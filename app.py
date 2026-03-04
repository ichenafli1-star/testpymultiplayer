import asyncio
import math
import os
import random
import string
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

DEFAULT_WORLD_WIDTH = 2400.0
DEFAULT_WORLD_HEIGHT = 1600.0
PLAYER_SIZE = 44.0
BULLET_RADIUS = 6.0
ROOM_TTL_SECONDS = 600
CLEANUP_INTERVAL_SECONDS = 30
TICK_RATE = 30
DISCONNECT_GRACE_SECONDS = 45.0
DISCONNECTED_PLAYER_DROP_SECONDS = 300.0


def clamp(value: float, mn: float, mx: float) -> float:
    return max(mn, min(mx, value))


@dataclass
class RoomConfig:
    world_width: float = DEFAULT_WORLD_WIDTH
    world_height: float = DEFAULT_WORLD_HEIGHT
    player_speed: float = 320.0
    bullet_speed: float = 900.0
    fire_cooldown: float = 0.22
    bullet_damage: int = 25
    max_hp: int = 100
    respawn_delay: float = 2.5


@dataclass
class InputState:
    left: bool = False
    right: bool = False
    up: bool = False
    down: bool = False
    shoot: bool = False


@dataclass
class Bullet:
    x: float
    y: float
    vx: float
    vy: float
    owner_id: str


@dataclass
class PlayerSlot:
    player_id: str
    name: str
    ws: Optional[WebSocket] = None
    connected: bool = False
    x: float = 0.0
    y: float = 0.0
    hp: int = 100
    alive: bool = True
    wins: int = 0
    fire_cd: float = 0.0
    facing_x: float = 1.0
    facing_y: float = 0.0
    input_state: InputState = field(default_factory=InputState)
    disconnected_at: Optional[float] = None


@dataclass
class Room:
    room_id: str
    owner_id: str
    config: RoomConfig = field(default_factory=RoomConfig)
    players: Dict[str, PlayerSlot] = field(default_factory=dict)
    bullets: list[Bullet] = field(default_factory=list)
    last_active: float = field(default_factory=time.time)
    round_over: bool = False
    reset_at: Optional[float] = None


class CreateRoomRequest(BaseModel):
    name: str = Field(default="Player 1", max_length=32)
    player_speed: float = Field(default=320.0, ge=120.0, le=1000.0)
    bullet_speed: float = Field(default=900.0, ge=200.0, le=2200.0)
    fire_cooldown: float = Field(default=0.22, ge=0.05, le=2.0)
    bullet_damage: int = Field(default=25, ge=1, le=100)
    max_hp: int = Field(default=100, ge=20, le=500)
    respawn_delay: float = Field(default=2.5, ge=1.0, le=10.0)


class JoinRoomRequest(BaseModel):
    room_id: str = Field(min_length=4, max_length=12)
    name: str = Field(default="Player", max_length=32)


app = FastAPI(title="Cube Tanks Online")
app.mount("/static", StaticFiles(directory="static"), name="static")

rooms: Dict[str, Room] = {}
rooms_lock = asyncio.Lock()
cleanup_task: Optional[asyncio.Task] = None
simulation_task: Optional[asyncio.Task] = None


def now_ts() -> float:
    return time.time()


def touch_room(room: Room) -> None:
    room.last_active = now_ts()


def generate_room_id() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


def set_player_spawn(room: Room, p: PlayerSlot, idx: int, total: int) -> None:
    cols = max(1, math.ceil(math.sqrt(total)))
    rows = max(1, math.ceil(total / cols))
    margin = 120.0
    world_w = room.config.world_width
    world_h = room.config.world_height
    space_x = (world_w - margin * 2) / max(1, cols - 1)
    space_y = (world_h - margin * 2) / max(1, rows - 1)
    r = idx // cols
    c = idx % cols
    p.x = margin + (space_x * c if cols > 1 else (world_w - PLAYER_SIZE) / 2)
    p.y = margin + (space_y * r if rows > 1 else (world_h - PLAYER_SIZE) / 2)
    p.x = min(max(0.0, p.x), world_w - PLAYER_SIZE)
    p.y = min(max(0.0, p.y), world_h - PLAYER_SIZE)


def spawn_new_player(room: Room, p: PlayerSlot) -> None:
    total = max(1, len(room.players))
    idx = max(0, total - 1)
    set_player_spawn(room, p, idx, total)
    p.hp = room.config.max_hp
    p.alive = True
    p.fire_cd = 0.0
    p.facing_x = 1.0 if idx % 2 == 0 else -1.0
    p.facing_y = 0.0
    p.input_state = InputState()
    p.disconnected_at = None


def reset_round(room: Room) -> None:
    ids = list(room.players.keys())
    total = len(ids)
    for i, pid in enumerate(ids):
        p = room.players[pid]
        set_player_spawn(room, p, i, max(1, total))
        p.hp = room.config.max_hp
        p.alive = True
        p.fire_cd = 0.0
        p.facing_x = 1.0 if i % 2 == 0 else -1.0
        p.facing_y = 0.0
        p.input_state = InputState()

    room.bullets.clear()
    room.round_over = False
    room.reset_at = None


def active_contender_count(room: Room) -> int:
    ts = now_ts()
    count = 0
    for p in room.players.values():
        if p.connected:
            count += 1
            continue
        if p.disconnected_at is not None and (ts - p.disconnected_at) <= DISCONNECT_GRACE_SECONDS:
            count += 1
    return count


def alive_contenders(room: Room) -> list[PlayerSlot]:
    ts = now_ts()
    result = []
    for p in room.players.values():
        if not p.alive:
            continue
        if p.connected:
            result.append(p)
            continue
        if p.disconnected_at is not None and (ts - p.disconnected_at) <= DISCONNECT_GRACE_SECONDS:
            result.append(p)
    return result


def build_state(room: Room, you: str) -> dict:
    players_payload = []
    ts = now_ts()
    for p in room.players.values():
        reconnect_left = None
        if not p.connected and p.disconnected_at is not None:
            reconnect_left = max(0.0, DISCONNECT_GRACE_SECONDS - (ts - p.disconnected_at))
        players_payload.append(
            {
                "player_id": p.player_id,
                "name": p.name,
                "connected": p.connected,
                "x": round(p.x, 2),
                "y": round(p.y, 2),
                "hp": p.hp,
                "alive": p.alive,
                "wins": p.wins,
                "facing_x": round(p.facing_x, 3),
                "facing_y": round(p.facing_y, 3),
                "reconnect_left": reconnect_left,
            }
        )

    bullets_payload = [{"x": round(b.x, 2), "y": round(b.y, 2)} for b in room.bullets]

    return {
        "type": "game_state",
        "room_id": room.room_id,
        "you": you,
        "owner_id": room.owner_id,
        "config": {
            "player_speed": room.config.player_speed,
            "bullet_speed": room.config.bullet_speed,
            "fire_cooldown": room.config.fire_cooldown,
            "bullet_damage": room.config.bullet_damage,
            "max_hp": room.config.max_hp,
            "respawn_delay": room.config.respawn_delay,
        },
        "world": {
            "width": room.config.world_width,
            "height": room.config.world_height,
            "player_size": PLAYER_SIZE,
        },
        "players": players_payload,
        "bullets": bullets_payload,
        "ready_to_play": active_contender_count(room) >= 2,
        "round_over": room.round_over,
    }


async def send_safe(ws: Optional[WebSocket], payload: dict) -> bool:
    if ws is None:
        return False
    try:
        await ws.send_json(payload)
        return True
    except Exception:
        return False


def mark_player_disconnected(room: Room, player_id: str) -> None:
    p = room.players.get(player_id)
    if p is None:
        return
    p.connected = False
    p.ws = None
    p.input_state = InputState()
    p.disconnected_at = now_ts()


def drop_stale_disconnected_players(room: Room) -> None:
    ts = now_ts()
    drop_ids = []
    for pid, p in room.players.items():
        if p.connected:
            continue
        if p.disconnected_at is None:
            continue
        if (ts - p.disconnected_at) > DISCONNECTED_PLAYER_DROP_SECONDS:
            drop_ids.append(pid)

    for pid in drop_ids:
        room.players.pop(pid, None)

    if room.owner_id not in room.players and room.players:
        room.owner_id = next(iter(room.players.keys()))


async def broadcast(room: Room, payload: dict):
    failed_ids = []
    for pid, p in room.players.items():
        if not p.connected:
            continue
        ok = await send_safe(p.ws, payload)
        if not ok:
            failed_ids.append(pid)

    for pid in failed_ids:
        mark_player_disconnected(room, pid)


async def broadcast_state(room: Room):
    failed_ids = []
    for pid, p in room.players.items():
        if not p.connected:
            continue
        ok = await send_safe(p.ws, build_state(room, pid))
        if not ok:
            failed_ids.append(pid)

    for pid in failed_ids:
        mark_player_disconnected(room, pid)


def update_player_topdown(room: Room, p: PlayerSlot, dt: float):
    if not p.alive:
        return

    dx = 0.0
    dy = 0.0
    if p.input_state.left:
        dx -= 1.0
    if p.input_state.right:
        dx += 1.0
    if p.input_state.up:
        dy -= 1.0
    if p.input_state.down:
        dy += 1.0

    if dx != 0.0 or dy != 0.0:
        ln = math.hypot(dx, dy)
        ux = dx / ln
        uy = dy / ln
        p.facing_x = ux
        p.facing_y = uy
        p.x += ux * room.config.player_speed * dt
        p.y += uy * room.config.player_speed * dt

    p.x = min(max(0.0, p.x), room.config.world_width - PLAYER_SIZE)
    p.y = min(max(0.0, p.y), room.config.world_height - PLAYER_SIZE)


def rect_hit(px: float, py: float, size: float, bx: float, by: float, radius: float) -> bool:
    nearest_x = max(px, min(bx, px + size))
    nearest_y = max(py, min(by, py + size))
    dx = bx - nearest_x
    dy = by - nearest_y
    return (dx * dx + dy * dy) <= (radius * radius)


def apply_owner_config_update(room: Room, raw: dict) -> None:
    room.config.player_speed = clamp(float(raw.get("player_speed", room.config.player_speed)), 120.0, 1000.0)
    room.config.bullet_speed = clamp(float(raw.get("bullet_speed", room.config.bullet_speed)), 200.0, 2200.0)
    room.config.fire_cooldown = clamp(float(raw.get("fire_cooldown", room.config.fire_cooldown)), 0.05, 2.0)
    room.config.bullet_damage = int(clamp(float(raw.get("bullet_damage", room.config.bullet_damage)), 1.0, 100.0))
    room.config.max_hp = int(clamp(float(raw.get("max_hp", room.config.max_hp)), 20.0, 500.0))
    room.config.respawn_delay = clamp(float(raw.get("respawn_delay", room.config.respawn_delay)), 1.0, 10.0)

    for p in room.players.values():
        p.hp = min(p.hp, room.config.max_hp)
        if p.hp <= 0:
            p.alive = False


def update_room_simulation(room: Room, dt: float) -> Optional[dict]:
    drop_stale_disconnected_players(room)

    if room.round_over:
        if room.reset_at is not None and now_ts() >= room.reset_at:
            reset_round(room)
            return {"type": "round_reset", "message": "Новый раунд начался"}
        return None

    for p in room.players.values():
        if not p.connected:
            continue

        p.fire_cd = max(0.0, p.fire_cd - dt)
        update_player_topdown(room, p, dt)

        if p.alive and p.input_state.shoot and p.fire_cd <= 0.0:
            dir_x = p.facing_x
            dir_y = p.facing_y
            if dir_x == 0.0 and dir_y == 0.0:
                dir_x = 1.0
            room.bullets.append(
                Bullet(
                    x=p.x + PLAYER_SIZE / 2.0 + dir_x * (PLAYER_SIZE / 2.0 + 8.0),
                    y=p.y + PLAYER_SIZE / 2.0 + dir_y * (PLAYER_SIZE / 2.0 + 8.0),
                    vx=dir_x * room.config.bullet_speed,
                    vy=dir_y * room.config.bullet_speed,
                    owner_id=p.player_id,
                )
            )
            p.fire_cd = room.config.fire_cooldown

    new_bullets: list[Bullet] = []
    for b in room.bullets:
        b.x += b.vx * dt
        b.y += b.vy * dt

        if (
            b.x < -12.0
            or b.x > room.config.world_width + 12.0
            or b.y < -12.0
            or b.y > room.config.world_height + 12.0
        ):
            continue

        hit = False
        for p in room.players.values():
            if not p.alive or p.player_id == b.owner_id:
                continue
            if rect_hit(p.x, p.y, PLAYER_SIZE, b.x, b.y, BULLET_RADIUS):
                p.hp = max(0, p.hp - room.config.bullet_damage)
                if p.hp == 0:
                    p.alive = False
                hit = True
                break

        if not hit:
            new_bullets.append(b)

    room.bullets = new_bullets

    contenders = active_contender_count(room)
    alive = alive_contenders(room)
    if contenders >= 2 and len(alive) <= 1:
        room.round_over = True
        room.reset_at = now_ts() + room.config.respawn_delay

        if len(alive) == 1:
            winner = alive[0]
            winner.wins += 1
            return {
                "type": "round_over",
                "winner_id": winner.player_id,
                "winner_name": winner.name,
                "message": f"Победил {winner.name}. Перезапуск через {room.config.respawn_delay:.1f} сек...",
            }

        return {
            "type": "round_over",
            "winner_id": None,
            "winner_name": None,
            "message": f"Ничья. Перезапуск через {room.config.respawn_delay:.1f} сек...",
        }

    return None


async def cleanup_stale_rooms_loop():
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        stale_ids = []
        async with rooms_lock:
            ts = now_ts()
            for room_id, room in rooms.items():
                connected_count = sum(1 for p in room.players.values() if p.connected)
                if connected_count == 0 and (ts - room.last_active) >= ROOM_TTL_SECONDS:
                    stale_ids.append(room_id)
            for room_id in stale_ids:
                rooms.pop(room_id, None)


async def simulation_loop():
    dt = 1.0 / TICK_RATE
    while True:
        await asyncio.sleep(dt)
        events: list[tuple[Room, dict]] = []
        active_rooms: list[Room] = []

        async with rooms_lock:
            for room in rooms.values():
                if not room.players:
                    continue
                touch_room(room)
                event = update_room_simulation(room, dt)
                active_rooms.append(room)
                if event:
                    events.append((room, event))

        for room, event in events:
            await broadcast(room, event)
        for room in active_rooms:
            await broadcast_state(room)


@app.on_event("startup")
async def on_startup():
    global cleanup_task, simulation_task
    if cleanup_task is None:
        cleanup_task = asyncio.create_task(cleanup_stale_rooms_loop())
    if simulation_task is None:
        simulation_task = asyncio.create_task(simulation_loop())


@app.on_event("shutdown")
async def on_shutdown():
    global cleanup_task, simulation_task
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        cleanup_task = None

    if simulation_task:
        simulation_task.cancel()
        try:
            await simulation_task
        except asyncio.CancelledError:
            pass
        simulation_task = None


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/api/create-room")
async def create_room(body: CreateRoomRequest):
    async with rooms_lock:
        room_id = generate_room_id()
        while room_id in rooms:
            room_id = generate_room_id()

        player_id = uuid.uuid4().hex
        config = RoomConfig(
            player_speed=body.player_speed,
            bullet_speed=body.bullet_speed,
            fire_cooldown=body.fire_cooldown,
            bullet_damage=body.bullet_damage,
            max_hp=body.max_hp,
            respawn_delay=body.respawn_delay,
        )
        room = Room(room_id=room_id, owner_id=player_id, config=config)
        p = PlayerSlot(player_id=player_id, name=body.name.strip() or "Player 1", connected=False)
        room.players[player_id] = p
        spawn_new_player(room, p)
        touch_room(room)
        rooms[room_id] = room

    return {
        "room_id": room_id,
        "player_id": player_id,
        "owner_id": player_id,
        "config": {
            "player_speed": config.player_speed,
            "bullet_speed": config.bullet_speed,
            "fire_cooldown": config.fire_cooldown,
            "bullet_damage": config.bullet_damage,
            "max_hp": config.max_hp,
            "respawn_delay": config.respawn_delay,
        },
    }


@app.post("/api/join-room")
async def join_room(body: JoinRoomRequest):
    room_id = body.room_id.strip().upper()

    async with rooms_lock:
        room = rooms.get(room_id)
        if room is None:
            raise HTTPException(status_code=404, detail="Комната не найдена")

        player_id = uuid.uuid4().hex
        p = PlayerSlot(player_id=player_id, name=body.name.strip() or "Player", connected=False)
        room.players[player_id] = p
        spawn_new_player(room, p)
        touch_room(room)

    return {
        "room_id": room_id,
        "player_id": player_id,
        "owner_id": room.owner_id,
        "config": {
            "player_speed": room.config.player_speed,
            "bullet_speed": room.config.bullet_speed,
            "fire_cooldown": room.config.fire_cooldown,
            "bullet_damage": room.config.bullet_damage,
            "max_hp": room.config.max_hp,
            "respawn_delay": room.config.respawn_delay,
        },
    }


@app.websocket("/ws/{room_id}")
async def game_ws(websocket: WebSocket, room_id: str, player_id: str):
    room_key = room_id.strip().upper()
    room: Optional[Room] = None
    await websocket.accept()

    async with rooms_lock:
        room = rooms.get(room_key)
        if room is None:
            await websocket.send_json({"type": "error", "message": "Комната не найдена"})
            await websocket.close(code=4404)
            return

        player = room.players.get(player_id)
        if player is None:
            await websocket.send_json({"type": "error", "message": "Неверный player_id"})
            await websocket.close(code=4403)
            return

        player.ws = websocket
        player.connected = True
        player.disconnected_at = None
        touch_room(room)

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            async with rooms_lock:
                room = rooms.get(room_key)
                if room is None or player_id not in room.players:
                    await websocket.send_json({"type": "error", "message": "Игрок не в комнате"})
                    continue

                p = room.players[player_id]

                if msg_type == "input":
                    p.input_state.left = bool(data.get("left", False))
                    p.input_state.right = bool(data.get("right", False))
                    p.input_state.up = bool(data.get("up", False))
                    p.input_state.down = bool(data.get("down", False))
                    p.input_state.shoot = bool(data.get("shoot", False))
                elif msg_type == "update_config":
                    if player_id != room.owner_id:
                        await websocket.send_json({"type": "error", "message": "Только создатель может менять настройки"})
                        continue
                    apply_owner_config_update(room, data)
                    await broadcast(room, {"type": "config_updated", "message": "Настройки обновлены"})
                else:
                    await websocket.send_json({"type": "error", "message": "Неподдерживаемый тип сообщения"})
                    continue

                touch_room(room)

    except WebSocketDisconnect:
        pass
    finally:
        async with rooms_lock:
            room = rooms.get(room_key)
            if room and player_id in room.players:
                mark_player_disconnected(room, player_id)
                touch_room(room)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host=host, port=port, reload=False)

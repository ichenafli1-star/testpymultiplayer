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

WORLD_WIDTH = 2400.0
WORLD_HEIGHT = 1600.0
PLAYER_SIZE = 44.0
PLAYER_SPEED = 320.0
MAX_HP = 100
BULLET_SPEED = 900.0
BULLET_RADIUS = 6.0
BULLET_DAMAGE = 25
FIRE_COOLDOWN = 0.22
ROOM_TTL_SECONDS = 600
CLEANUP_INTERVAL_SECONDS = 30
TICK_RATE = 30


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
    hp: int = MAX_HP
    alive: bool = True
    wins: int = 0
    fire_cd: float = 0.0
    facing_x: float = 1.0
    facing_y: float = 0.0
    input_state: InputState = field(default_factory=InputState)


@dataclass
class Room:
    room_id: str
    players: Dict[str, PlayerSlot] = field(default_factory=dict)
    bullets: list[Bullet] = field(default_factory=list)
    last_active: float = field(default_factory=time.time)
    round_over: bool = False
    reset_at: Optional[float] = None


class CreateRoomRequest(BaseModel):
    name: str = Field(default="Player 1", max_length=32)


class JoinRoomRequest(BaseModel):
    room_id: str = Field(min_length=4, max_length=12)
    name: str = Field(default="Player 2", max_length=32)


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


def reset_online_players(room: Room) -> None:
    online_ids = [pid for pid, p in room.players.items() if p.connected]
    if not online_ids:
        room.bullets.clear()
        room.round_over = False
        room.reset_at = None
        return

    n = len(online_ids)
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = max(1, math.ceil(n / cols))
    margin = 120.0
    space_x = (WORLD_WIDTH - margin * 2) / max(1, cols - 1)
    space_y = (WORLD_HEIGHT - margin * 2) / max(1, rows - 1)

    for i, pid in enumerate(online_ids):
        p = room.players[pid]
        r = i // cols
        c = i % cols
        p.x = margin + (space_x * c if cols > 1 else (WORLD_WIDTH - PLAYER_SIZE) / 2)
        p.y = margin + (space_y * r if rows > 1 else (WORLD_HEIGHT - PLAYER_SIZE) / 2)
        p.x = min(max(0.0, p.x), WORLD_WIDTH - PLAYER_SIZE)
        p.y = min(max(0.0, p.y), WORLD_HEIGHT - PLAYER_SIZE)
        p.hp = MAX_HP
        p.alive = True
        p.fire_cd = 0.0
        p.facing_x = 1.0 if i % 2 == 0 else -1.0
        p.facing_y = 0.0
        p.input_state = InputState()

    room.bullets.clear()
    room.round_over = False
    room.reset_at = None


def build_state(room: Room, you: str) -> dict:
    players_payload = []
    for p in room.players.values():
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
            }
        )

    bullets_payload = [{"x": round(b.x, 2), "y": round(b.y, 2)} for b in room.bullets]
    online_count = sum(1 for p in room.players.values() if p.connected)

    return {
        "type": "game_state",
        "room_id": room.room_id,
        "you": you,
        "world": {
            "width": WORLD_WIDTH,
            "height": WORLD_HEIGHT,
            "player_size": PLAYER_SIZE,
        },
        "players": players_payload,
        "bullets": bullets_payload,
        "ready_to_play": online_count >= 2,
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


async def broadcast(room: Room, payload: dict):
    for p in room.players.values():
        if p.connected:
            ok = await send_safe(p.ws, payload)
            if not ok:
                p.connected = False
                p.ws = None


async def broadcast_state(room: Room):
    for pid, p in room.players.items():
        if p.connected:
            ok = await send_safe(p.ws, build_state(room, pid))
            if not ok:
                p.connected = False
                p.ws = None


def update_player_topdown(p: PlayerSlot, dt: float):
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
        p.x += ux * PLAYER_SPEED * dt
        p.y += uy * PLAYER_SPEED * dt

    p.x = min(max(0.0, p.x), WORLD_WIDTH - PLAYER_SIZE)
    p.y = min(max(0.0, p.y), WORLD_HEIGHT - PLAYER_SIZE)


def rect_hit(px: float, py: float, size: float, bx: float, by: float, radius: float) -> bool:
    nearest_x = max(px, min(bx, px + size))
    nearest_y = max(py, min(by, py + size))
    dx = bx - nearest_x
    dy = by - nearest_y
    return (dx * dx + dy * dy) <= (radius * radius)


def update_room_simulation(room: Room, dt: float) -> Optional[dict]:
    if room.round_over:
        if room.reset_at is not None and now_ts() >= room.reset_at:
            reset_online_players(room)
            return {"type": "round_reset", "message": "Новый раунд начался"}
        return None

    for p in room.players.values():
        if p.connected:
            p.fire_cd = max(0.0, p.fire_cd - dt)
            update_player_topdown(p, dt)

            if p.alive and p.input_state.shoot and p.fire_cd <= 0.0:
                dir_x = p.facing_x
                dir_y = p.facing_y
                if dir_x == 0.0 and dir_y == 0.0:
                    dir_x = 1.0
                room.bullets.append(
                    Bullet(
                        x=p.x + PLAYER_SIZE / 2.0 + dir_x * (PLAYER_SIZE / 2.0 + 8.0),
                        y=p.y + PLAYER_SIZE / 2.0 + dir_y * (PLAYER_SIZE / 2.0 + 8.0),
                        vx=dir_x * BULLET_SPEED,
                        vy=dir_y * BULLET_SPEED,
                        owner_id=p.player_id,
                    )
                )
                p.fire_cd = FIRE_COOLDOWN

    new_bullets: list[Bullet] = []
    for b in room.bullets:
        b.x += b.vx * dt
        b.y += b.vy * dt

        if b.x < -12.0 or b.x > WORLD_WIDTH + 12.0 or b.y < -12.0 or b.y > WORLD_HEIGHT + 12.0:
            continue

        hit = False
        for p in room.players.values():
            if not p.connected or not p.alive or p.player_id == b.owner_id:
                continue
            if rect_hit(p.x, p.y, PLAYER_SIZE, b.x, b.y, BULLET_RADIUS):
                p.hp = max(0, p.hp - BULLET_DAMAGE)
                if p.hp == 0:
                    p.alive = False
                hit = True
                break

        if not hit:
            new_bullets.append(b)

    room.bullets = new_bullets

    online_players = [p for p in room.players.values() if p.connected]
    alive_online = [p for p in online_players if p.alive]

    if len(online_players) >= 2 and len(alive_online) == 1:
        winner = alive_online[0]
        winner.wins += 1
        room.round_over = True
        room.reset_at = now_ts() + 2.5
        return {
            "type": "round_over",
            "winner_id": winner.player_id,
            "winner_name": winner.name,
            "message": f"Победил {winner.name}. Перезапуск через 2.5 сек...",
        }

    if len(online_players) >= 2 and len(alive_online) == 0:
        room.round_over = True
        room.reset_at = now_ts() + 2.5
        return {
            "type": "round_over",
            "winner_id": None,
            "winner_name": None,
            "message": "Ничья. Перезапуск через 2.5 сек...",
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
                connected_count = sum(1 for p in room.players.values() if p.connected)
                if connected_count == 0:
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
        room = Room(room_id=room_id)
        room.players[player_id] = PlayerSlot(player_id=player_id, name=body.name.strip() or "Player 1")
        reset_online_players(room)
        touch_room(room)
        rooms[room_id] = room

    return {"room_id": room_id, "player_id": player_id}


@app.post("/api/join-room")
async def join_room(body: JoinRoomRequest):
    room_id = body.room_id.strip().upper()

    async with rooms_lock:
        room = rooms.get(room_id)
        if room is None:
            raise HTTPException(status_code=404, detail="Комната не найдена")

        player_id = uuid.uuid4().hex
        room.players[player_id] = PlayerSlot(player_id=player_id, name=body.name.strip() or "Player")
        reset_online_players(room)
        touch_room(room)

    return {"room_id": room_id, "player_id": player_id}


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
        reset_online_players(room)
        touch_room(room)

    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") != "input":
                await websocket.send_json({"type": "error", "message": "Неподдерживаемый тип сообщения"})
                continue

            async with rooms_lock:
                room = rooms.get(room_key)
                if room is None or player_id not in room.players:
                    await websocket.send_json({"type": "error", "message": "Игрок не в комнате"})
                    continue

                p = room.players[player_id]
                p.input_state.left = bool(data.get("left", False))
                p.input_state.right = bool(data.get("right", False))
                p.input_state.up = bool(data.get("up", False))
                p.input_state.down = bool(data.get("down", False))
                p.input_state.shoot = bool(data.get("shoot", False))
                touch_room(room)

    except WebSocketDisconnect:
        pass
    finally:
        async with rooms_lock:
            room = rooms.get(room_key)
            if room and player_id in room.players:
                p = room.players[player_id]
                p.connected = False
                p.ws = None
                p.input_state = InputState()
                touch_room(room)
                reset_online_players(room)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host=host, port=port, reload=False)

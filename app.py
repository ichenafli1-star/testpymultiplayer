import asyncio
import os
import random
import string
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

MOVES = {"rock", "paper", "scissors"}
WIN_MAP = {
    "rock": "scissors",
    "scissors": "paper",
    "paper": "rock",
}
ROOM_TTL_SECONDS = 600
CLEANUP_INTERVAL_SECONDS = 30


@dataclass
class PlayerSlot:
    player_id: str
    name: str
    ws: Optional[WebSocket] = None
    connected: bool = False
    move: Optional[str] = None
    score: int = 0


@dataclass
class Room:
    room_id: str
    target_score: int
    players: Dict[str, PlayerSlot] = field(default_factory=dict)
    rematch_requests: Set[str] = field(default_factory=set)
    match_winner: Optional[str] = None
    last_active: float = field(default_factory=time.time)


class CreateRoomRequest(BaseModel):
    name: str = Field(default="Player 1", max_length=32)
    target_score: int = Field(default=3, ge=1, le=15)


class JoinRoomRequest(BaseModel):
    room_id: str = Field(min_length=4, max_length=12)
    name: str = Field(default="Player 2", max_length=32)


app = FastAPI(title="Rock Paper Scissors Online")
app.mount("/static", StaticFiles(directory="static"), name="static")

rooms: Dict[str, Room] = {}
rooms_lock = asyncio.Lock()
cleanup_task: Optional[asyncio.Task] = None


def now_ts() -> float:
    return time.time()


def touch_room(room: Room) -> None:
    room.last_active = now_ts()


def generate_room_id() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


def display_name(room: Room, player_id: str) -> str:
    participant = room.players.get(player_id)
    if participant is None:
        return player_id
    return participant.name


def make_state(room: Room, you: str) -> dict:
    players = [
        {
            "player_id": p.player_id,
            "name": p.name,
            "connected": p.connected,
            "score": p.score,
            "has_move": p.move is not None,
        }
        for p in room.players.values()
    ]
    connected_count = sum(1 for p in room.players.values() if p.connected)
    return {
        "type": "state",
        "room_id": room.room_id,
        "you": you,
        "players": players,
        "target_score": room.target_score,
        "match_winner": room.match_winner,
        "ready_to_play": connected_count == 2 and len(room.players) == 2 and room.match_winner is None,
        "rematch_requested_by": list(room.rematch_requests),
    }


def decide_winner(p1_id: str, p1_move: str, p2_id: str, p2_move: str) -> Optional[str]:
    if p1_move == p2_move:
        return None
    if WIN_MAP[p1_move] == p2_move:
        return p1_id
    return p2_id


async def send_safe(ws: Optional[WebSocket], payload: dict) -> bool:
    if ws is None:
        return False
    try:
        await ws.send_json(payload)
        return True
    except Exception:
        return False


async def broadcast_state(room: Room):
    for pid, player in room.players.items():
        if player.connected:
            ok = await send_safe(player.ws, make_state(room, pid))
            if not ok:
                player.connected = False
                player.ws = None
                player.move = None


async def cleanup_stale_rooms_loop():
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        stale_ids = []
        async with rooms_lock:
            ts = now_ts()
            for room_id, room in rooms.items():
                connected_count = sum(1 for p in room.players.values() if p.connected)
                stale = connected_count == 0 and (ts - room.last_active) >= ROOM_TTL_SECONDS
                if stale:
                    stale_ids.append(room_id)

            for room_id in stale_ids:
                rooms.pop(room_id, None)


@app.on_event("startup")
async def on_startup():
    global cleanup_task
    if cleanup_task is None:
        cleanup_task = asyncio.create_task(cleanup_stale_rooms_loop())


@app.on_event("shutdown")
async def on_shutdown():
    global cleanup_task
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        cleanup_task = None


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
        room = Room(room_id=room_id, target_score=body.target_score)
        room.players[player_id] = PlayerSlot(player_id=player_id, name=body.name.strip() or "Player 1")
        touch_room(room)
        rooms[room_id] = room

    return {"room_id": room_id, "player_id": player_id, "target_score": room.target_score}


@app.post("/api/join-room")
async def join_room(body: JoinRoomRequest):
    room_id = body.room_id.strip().upper()

    async with rooms_lock:
        room = rooms.get(room_id)
        if room is None:
            raise HTTPException(status_code=404, detail="Комната не найдена")

        if len(room.players) >= 2:
            raise HTTPException(status_code=400, detail="Комната уже заполнена")

        player_id = uuid.uuid4().hex
        room.players[player_id] = PlayerSlot(player_id=player_id, name=body.name.strip() or "Player 2")
        touch_room(room)

    await broadcast_state(room)
    return {"room_id": room_id, "player_id": player_id, "target_score": room.target_score}


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
        touch_room(room)

    await broadcast_state(room)

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            round_payload = None
            match_payload = None
            info_payload = None

            async with rooms_lock:
                if room is None or player_id not in room.players:
                    await websocket.send_json({"type": "error", "message": "Игрок не в комнате"})
                    continue

                current = room.players[player_id]
                if not current.connected:
                    await websocket.send_json({"type": "error", "message": "Нет активного подключения"})
                    continue

                touch_room(room)

                if msg_type == "move":
                    move = str(data.get("move", "")).lower().strip()
                    if move not in MOVES:
                        await websocket.send_json({"type": "error", "message": "Недопустимый ход"})
                        continue

                    if room.match_winner is not None:
                        await websocket.send_json({"type": "error", "message": "Матч завершен. Нажмите реванш"})
                        continue

                    if current.move is not None:
                        await websocket.send_json({"type": "error", "message": "Ход уже отправлен"})
                        continue

                    current.move = move

                    if len(room.players) != 2:
                        info_payload = {"type": "info", "message": "Ожидание второго игрока"}
                    else:
                        pids = list(room.players.keys())
                        p1 = room.players[pids[0]]
                        p2 = room.players[pids[1]]

                        if p1.move is not None and p2.move is not None:
                            winner = decide_winner(p1.player_id, p1.move, p2.player_id, p2.move)
                            if winner:
                                room.players[winner].score += 1

                            round_payload = {
                                "type": "round_result",
                                "moves": {
                                    p1.player_id: p1.move,
                                    p2.player_id: p2.move,
                                },
                                "winner": winner,
                            }

                            if winner and room.players[winner].score >= room.target_score:
                                room.match_winner = winner
                                room.rematch_requests.clear()
                                match_payload = {
                                    "type": "match_over",
                                    "winner": winner,
                                    "winner_name": display_name(room, winner),
                                    "target_score": room.target_score,
                                }

                            p1.move = None
                            p2.move = None

                elif msg_type == "rematch_request":
                    if room.match_winner is None:
                        await websocket.send_json({"type": "error", "message": "Матч еще не завершен"})
                        continue

                    room.rematch_requests.add(player_id)

                    if len(room.rematch_requests) == len(room.players) == 2:
                        for participant in room.players.values():
                            participant.score = 0
                            participant.move = None
                        room.match_winner = None
                        room.rematch_requests.clear()
                        match_payload = {"type": "match_reset", "message": "Новый матч начался"}
                    else:
                        info_payload = {"type": "info", "message": "Запрос реванша отправлен"}

                else:
                    await websocket.send_json({"type": "error", "message": "Неподдерживаемый тип сообщения"})
                    continue

            if info_payload:
                await send_safe(websocket, info_payload)

            if room and round_payload:
                for participant in room.players.values():
                    await send_safe(participant.ws, round_payload)

            if room and match_payload:
                for participant in room.players.values():
                    await send_safe(participant.ws, match_payload)

            if room:
                await broadcast_state(room)

    except WebSocketDisconnect:
        pass
    finally:
        async with rooms_lock:
            room = rooms.get(room_key)
            if room and player_id in room.players:
                participant = room.players[player_id]
                participant.connected = False
                participant.ws = None
                participant.move = None
                touch_room(room)

        if room:
            await broadcast_state(room)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host=host, port=port, reload=False)

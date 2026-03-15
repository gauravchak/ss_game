import { NextResponse } from 'next/server';
import { Redis } from '@upstash/redis';

// Safely initialize Redis so local dev doesn't crash if env vars are missing
let redis: Redis | null = null;
try {
  if (process.env.UPSTASH_REDIS_REST_URL && process.env.UPSTASH_REDIS_REST_TOKEN) {
    redis = Redis.fromEnv();
  }
} catch (e) {
  console.warn("Upstash Redis env vars missing. Caching disabled for hints.");
}

type Position = { r: number; c: number };
type Move = { r1: number; c1: number; r2: number; c2: number };
type BoardState = ('O' | '.' | ' ')[][];

// Center position for a win in standard English Peg Solitaire
const CENTER_R = 3;
const CENTER_C = 3;

// Directions: Right, Left, Down, Up
const DIRS = [[0, 2], [0, -2], [2, 0], [-2, 0]];

export async function POST(request: Request) {
  try {
    const { board } = await request.json();
    if (!board || !Array.isArray(board)) {
      return NextResponse.json({ error: 'Invalid board state' }, { status: 400 });
    }

    const stateString = boardToCompactString(board);

    // 1. Check Redis Cache
    let cachedMove: Move | 'UNSOLVABLE' | null = null;
    if (redis) {
        cachedMove = await redis.get<Move | 'UNSOLVABLE'>(`hint:${stateString}`);
    }
    
    if (cachedMove !== null) {
      if (cachedMove === 'UNSOLVABLE') {
        return NextResponse.json({ hint: null, message: 'No winning sequence exists from this position' });
      }
      return NextResponse.json({ hint: cachedMove, source: 'cache' });
    }

    // 2. Not cached. Check if it's already a win or a blatant loss
    const pegs = countPegs(board);
    if (pegs === 1 && board[CENTER_R][CENTER_C] === 'O') {
        return NextResponse.json({ hint: null, message: 'Game already won!' });
    }

    // Security/Performance measure: Deep DFS on Vercel Edge functions can timeout.
    // Full 32-peg board takes minutes/hours in raw JS DFS.
    // If pegs > 15, we do a very shallow greedy search just to give *any* valid move that keeps the game alive,
    // or we cap the DFS depth. For a true hint system, doing a full DFS online is best when pegs < 15.
    
    // We will run a depth-limited or bounded DFS here, but for Peg Solitaire on a Vercel function,
    // we need to be careful with execution time (max 10-60s depending on plan).
    // Let's implement the DFS but restrict the node exploration count.
    
    const startTime = Date.now();
    const result = findWinningMove(board, new Set<string>(), 0, 50000); // Max 50k states explored
    
    if (result) {
        // Cache the successful first move
        if (redis) {
            await redis.set(`hint:${stateString}`, result, { ex: 60 * 60 * 24 * 7 }); // Cache 7 days
        }
        return NextResponse.json({ hint: result, source: 'compute', timeMs: Date.now() - startTime });
    } else {
        // We either exhausted search (unsolvable) or hit the node limit.
        // If we hit the limit, we shouldn't cache as 'UNSOLVABLE' because it might be solvable.
        // Let's just return a random valid move as a "fallback hint" if we timeout.
        const validMoves = getAllValidMoves(board);
        if (validMoves.length > 0) {
           const fallback = validMoves[Math.floor(Math.random() * validMoves.length)];
           return NextResponse.json({ 
             hint: fallback, 
             source: 'fallback_random',
             message: 'Board too complex for deep search. Here is a valid random move.' 
           });
        }

        // Truly unsolvable (no moves left)
        if (redis) {
            await redis.set(`hint:${stateString}`, 'UNSOLVABLE', { ex: 60 * 60 * 24 });
        }
        return NextResponse.json({ hint: null, message: 'No valid moves exist.' });
    }

  } catch (err) {
    console.error('Hint API Error:', err);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}

// --- DFS Logic ---

function findWinningMove(
    board: BoardState, 
    visited: Set<string>, 
    depth: number, 
    maxNodes: number
): Move | null {
    // DFS State tracker Hack (using a mutable object to track nodes across recursive calls)
    const tracer = { nodes: 0 };
    
    const dfs = (b: BoardState): Move | null => {
        tracer.nodes++;
        if (tracer.nodes > maxNodes) return null; // Abort if taking too long

        const stateStr = boardToCompactString(b);
        if (visited.has(stateStr)) return null;
        visited.add(stateStr);

        const pegs = countPegs(b);
        if (pegs === 1) {
            return b[CENTER_R][CENTER_C] === 'O' ? { r1:-1, c1:-1, r2:-1, c2:-1 } : null; // Won!
        }

        const moves = getAllValidMoves(b);
        if (moves.length === 0) return null; // Dead end

        for (const move of moves) {
             const nextBoard = applyMove(b, move);
             const res = dfs(nextBoard);
             if (res !== null) {
                 // If the child call found a path to victory, WE return THIS move
                 // as the correct step to take from the current board.
                 return move; 
             }
        }

        return null;
    };

    return dfs(board);
}

function countPegs(board: BoardState): number {
    let count = 0;
    for(let r=0; r<7; r++) {
        for(let c=0; c<7; c++) {
            if(board[r][c] === 'O') count++;
        }
    }
    return count;
}

function getAllValidMoves(board: BoardState): Move[] {
    const moves: Move[] = [];
    for(let r=0; r<7; r++) {
        for(let c=0; c<7; c++) {
            if (board[r][c] === 'O') {
                for (const [dr, dc] of DIRS) {
                    const r2 = r + dr;
                    const c2 = c + dc;
                    const midR = r + dr/2;
                    const midC = c + dc/2;
                    
                    if (r2>=0 && r2<7 && c2>=0 && c2<7 && 
                        board[r2][c2] === '.' && 
                        board[midR][midC] === 'O') {
                        moves.push({ r1: r, c1: c, r2, c2 });
                    }
                }
            }
        }
    }
    // Optimization: Sort moves heuristically (e.g. prioritize middle-bound moves)
    // to find the winning path faster in DFS.
    moves.sort((a, b) => {
        const distA = Math.abs(a.r2 - CENTER_R) + Math.abs(a.c2 - CENTER_C);
        const distB = Math.abs(b.r2 - CENTER_R) + Math.abs(b.c2 - CENTER_C);
        return distA - distB; 
    });

    return moves;
}

function applyMove(board: BoardState, move: Move): BoardState {
    const newBoard = board.map(row => [...row]);
    newBoard[move.r1][move.c1] = '.';
    newBoard[move.r1 + (move.r2 - move.r1)/2][move.c1 + (move.c2 - move.c1)/2] = '.';
    newBoard[move.r2][move.c2] = 'O';
    return newBoard;
}

// Compact string logic to save cache memory "OOOOOOO...OO...O "
function boardToCompactString(board: BoardState): string {
    let str = "";
    for(let r=0; r<7; r++) {
        for(let c=0; c<7; c++) {
            str += board[r][c];
        }
    }
    return str;
}

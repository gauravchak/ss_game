'use server';

import { Redis } from '@upstash/redis';

// Initialize the Redis client. 
// When deployed to Vercel and linked with Upstash Redis, these env vars are auto-populated.
const redis = Redis.fromEnv();

export interface ScoreEntry {
  id: string; // KV doesn't auto-increment easily, we'll use a timestamp-based ID
  player_name: string;
  marbles_remaining: number;
  created_at: string;
}

const LEADERBOARD_KEY = 'solitaire:leaderboard';

export async function submitScoreAction(playerName: string, marblesRemaining: number) {
  try {
    const newEntry: ScoreEntry = {
      id: Date.now().toString(),
      player_name: playerName.slice(0, 20).trim(), // Protect against massive string inputs
      marbles_remaining: marblesRemaining,
      created_at: new Date().toISOString(),
    };

    // 1. Fetch current leaderboard
    let currentLeaderboard = await redis?.get<ScoreEntry[]>(LEADERBOARD_KEY) || [];
    
    // 2. Add new score
    currentLeaderboard.push(newEntry);
    
    // 3. Sort (least marbles first, then newest first)
    currentLeaderboard.sort((a, b) => {
      if (a.marbles_remaining !== b.marbles_remaining) {
        return a.marbles_remaining - b.marbles_remaining;
      }
      return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
    });
    
    // 4. Keep top 100 to save space but allow for client-side aggregation
    currentLeaderboard = currentLeaderboard.slice(0, 100);
    
    // 5. Save back to KV
    await redis?.set(LEADERBOARD_KEY, currentLeaderboard);
    
    return { success: true };
  } catch (error) {
    console.error('Failed to submit score to Redis:', error);
    return { success: false, error: 'Failed to submit score' };
  }
}

export async function getLeaderboardAction(): Promise<ScoreEntry[]> {
  try {
    const data = await redis?.get<ScoreEntry[]>(LEADERBOARD_KEY);
    return data || [];
  } catch (error) {
    console.error('Failed to fetch leaderboard from Redis:', error);
    return [];
  }
}

// --- RL Data Collection ---
export interface TrajectoryEntry {
  id: string;
  outcome: number; // marbles remaining
  timestamp: string;
  moves: {
    state: string; // 49-char compact string representation of the board
    action: { row: number; col: number; dir: number }; // RL optimized action space
  }[];
}

export async function saveTrajectoryAction(outcome: number, moves: TrajectoryEntry['moves']) {
  if (!redis) return { success: false, error: 'Redis client not configured' };
  
  // We only want to save games that actually had moves, 
  // ignoring if someone loaded the page and immediately quit/won somehow.
  if (moves.length === 0) return { success: false, error: 'No moves to save' };

  try {
    const trajectory: TrajectoryEntry = {
      id: Date.now().toString() + '_' + Math.random().toString(36).substr(2, 5),
      outcome,
      timestamp: new Date().toISOString(),
      moves
    };

    // Push raw trajectory to the right end of a Redis List
    await redis.rpush('rl:trajectories', JSON.stringify(trajectory));
    return { success: true };
  } catch (error) {
    console.error('Failed to save trajectory to Redis:', error);
    return { success: false, error: 'Failed to save trajectory' };
  }
}

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
      player_name: playerName.slice(0, 20), // Protect against massive string inputs
      marbles_remaining: marblesRemaining,
      created_at: new Date().toISOString(),
    };

    // 1. Fetch current leaderboard
    let currentLeaderboard = await redis.get<ScoreEntry[]>(LEADERBOARD_KEY) || [];
    
    // 2. Add new score
    currentLeaderboard.push(newEntry);
    
    // 3. Sort (least marbles first, then newest first)
    currentLeaderboard.sort((a, b) => {
      if (a.marbles_remaining !== b.marbles_remaining) {
        return a.marbles_remaining - b.marbles_remaining;
      }
      return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
    });
    
    // 4. Keep only top 10 to save space
    currentLeaderboard = currentLeaderboard.slice(0, 10);
    
    // 5. Save back to KV
    await redis.set(LEADERBOARD_KEY, currentLeaderboard);
    
    return { success: true };
  } catch (error) {
    console.error('Failed to submit score to Redis:', error);
    return { success: false, error: 'Failed to submit score' };
  }
}

export async function getLeaderboardAction(): Promise<ScoreEntry[]> {
  try {
    const data = await redis.get<ScoreEntry[]>(LEADERBOARD_KEY);
    return data || [];
  } catch (error) {
    console.error('Failed to fetch leaderboard from Redis:', error);
    return [];
  }
}

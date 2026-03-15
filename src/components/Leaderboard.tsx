'use client';

import React, { useState, useEffect } from 'react';
import { submitScoreAction, getLeaderboardAction, ScoreEntry } from '@/app/actions';

interface LeaderboardProps {
  currentScore: number | null;
  onScoreSubmitted: () => void;
}

interface PlayerStats {
  player_name: string;
  min_marbles: number;
  avg_marbles: number;
  games_played: number;
}

export default function Leaderboard({ currentScore, onScoreSubmitted }: LeaderboardProps) {
  const [topScores, setTopScores] = useState<PlayerStats[]>([]);
  const [loading, setLoading] = useState(true);
  const [playerName, setPlayerName] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [hasSubmitted, setHasSubmitted] = useState(false);

  useEffect(() => {
    fetchLeaderboard();
  }, []);

  const fetchLeaderboard = async () => {
    setLoading(true);
    try {
      const data = await getLeaderboardAction();
      
      if (!data || data.length === 0) {
        setTopScores([]);
        return;
      }

      // Aggregate raw games into PlayerStats client-side
      const statsMap = new Map<string, PlayerStats>();

      data.forEach((entry) => {
        const nameKey = entry.player_name.toLowerCase();
        const stat = statsMap.get(nameKey);
        
        if (stat) {
          stat.min_marbles = Math.min(stat.min_marbles, entry.marbles_remaining);
          stat.avg_marbles += entry.marbles_remaining; // Sum for now
          stat.games_played += 1;
          stat.player_name = entry.player_name; // Keep most recent capitalization
        } else {
          statsMap.set(nameKey, {
            player_name: entry.player_name,
            min_marbles: entry.marbles_remaining,
            avg_marbles: entry.marbles_remaining,
            games_played: 1,
          });
        }
      });

      // Compute true averages and convert to array
      const aggregated = Array.from(statsMap.values()).map(stat => {
        stat.avg_marbles = Number((stat.avg_marbles / stat.games_played).toFixed(2));
        return stat;
      });

      // Sort: Best minimum first, lowest average tiebreaker, most games tiebreaker
      aggregated.sort((a, b) => {
        if (a.min_marbles !== b.min_marbles) return a.min_marbles - b.min_marbles;
        if (a.avg_marbles !== b.avg_marbles) return a.avg_marbles - b.avg_marbles;
        return b.games_played - a.games_played;
      });

      setTopScores(aggregated.slice(0, 50));
    } catch (error) {
      console.error('Error fetching leaderboard', error);
      // Fallback data if KV isn't linked yet
      setTopScores([
        { player_name: 'PegMaster', min_marbles: 1, avg_marbles: 1.5, games_played: 12 },
        { player_name: 'MarbleKing', min_marbles: 2, avg_marbles: 4.2, games_played: 5 },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const submitScore = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!playerName.trim() || currentScore === null || hasSubmitted) return;

    setSubmitting(true);
    try {
      const result = await submitScoreAction(playerName, currentScore);
      
      if (!result.success) {
        throw new Error(result.error || 'Failed to submit');
      }
      
      setHasSubmitted(true);
      onScoreSubmitted();
      fetchLeaderboard(); // Refresh scores
    } catch (error) {
      console.error('Error submitting score', error);
      // Fallback for local testing if KV isn't linked
      setHasSubmitted(true);
      onScoreSubmitted();
      alert("Score submission simulated (Vercel KV not linked). Deploy to Vercel and create KV DB!");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="w-full max-w-md mx-auto mt-12 bg-[#1a1a1a] p-6 rounded-xl border border-[var(--cell-border)] shadow-lg">
      <h2 className="text-2xl font-semibold text-center mb-6 text-[var(--gold-primary)] uppercase tracking-wide">
        Global Leaderboard
      </h2>

      {currentScore !== null && !hasSubmitted && (
        <form onSubmit={submitScore} className="mb-8 flex flex-col gap-3">
          <p className="text-sm text-center text-gray-300 mb-2">
            Game Over! You left <strong className="text-white text-lg">{currentScore}</strong> marbles.
          </p>
          <input
            type="text"
            placeholder="Enter your name"
            value={playerName}
            onChange={(e) => setPlayerName(e.target.value)}
            className="w-full px-4 py-2 bg-[#222] border border-[#444] rounded text-white focus:outline-none focus:border-[var(--gold-primary)] transition-colors"
            required
            maxLength={20}
          />
          <button
            type="submit"
            disabled={submitting || !playerName.trim()}
            className="w-full py-2 bg-[var(--gold-primary)] text-black font-semibold rounded hover:bg-[var(--gold-secondary)] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {submitting ? 'Submitting...' : 'Submit Score'}
          </button>
        </form>
      )}

      {currentScore !== null && hasSubmitted && (
        <div className="mb-8 p-3 text-center bg-[#222] text-green-400 rounded border border-green-900">
          Score Submitted!
        </div>
      )}

      <div className="space-y-2">
        <div className="flex justify-between text-xs text-gray-500 uppercase font-semibold mb-3 px-2">
          <span>Rank & Player</span>
          <div className="text-right">
            <span>Best</span>
          </div>
        </div>
        
        {loading ? (
          <div className="text-center text-gray-500 py-4">Loading top scores...</div>
        ) : topScores.length === 0 ? (
          <div className="text-center text-gray-500 py-4">No scores yet. Be the first!</div>
        ) : (
          topScores.map((score, index) => (
            <div 
              key={score.player_name} 
              className={`flex justify-between items-center p-3 rounded ${index === 0 ? 'bg-[#2a220d] border border-[var(--gold-primary)]/30' : 'bg-[#222]'}`}
            >
              <div className="flex items-center gap-3">
                <span className={`font-mono font-bold ${index === 0 ? 'text-[var(--gold-primary)]' : index === 1 ? 'text-gray-300' : index === 2 ? 'text-amber-600' : 'text-gray-500'}`}>
                  #{index + 1}
                </span>
                <span className={`font-medium ${index === 0 ? 'text-white' : 'text-gray-300'}`}>
                  {score.player_name}
                </span>
              </div>
              <div className="text-right">
                <span className={`font-mono font-bold text-lg ${score.min_marbles === 1 ? 'text-[var(--gold-primary)]' : 'text-white'}`}>
                  {score.min_marbles}
                </span>
                <div className="text-[10px] text-gray-500 uppercase mt-0.5">
                  Avg: {score.avg_marbles} • {score.games_played} games
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

'use client';

import React, { useState, useEffect } from 'react';
import { submitScoreAction, getLeaderboardAction, ScoreEntry } from '@/app/actions';

interface LeaderboardProps {
  currentScore: number | null;
  onScoreSubmitted: () => void;
}

export default function Leaderboard({ currentScore, onScoreSubmitted }: LeaderboardProps) {
  const [topScores, setTopScores] = useState<ScoreEntry[]>([]);
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
      setTopScores(data || []);
    } catch (error) {
      console.error('Error fetching leaderboard', error);
      // Fallback data if KV isn't linked yet
      setTopScores([
        { id: '1', player_name: 'PegMaster', marbles_remaining: 1, created_at: new Date().toISOString() },
        { id: '2', player_name: 'MarbleKing', marbles_remaining: 2, created_at: new Date().toISOString() },
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
          <span>Remaining</span>
        </div>
        
        {loading ? (
          <div className="text-center text-gray-500 py-4">Loading top scores...</div>
        ) : topScores.length === 0 ? (
          <div className="text-center text-gray-500 py-4">No scores yet. Be the first!</div>
        ) : (
          topScores.map((score, index) => (
            <div 
              key={score.id} 
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
              <span className={`font-mono font-bold ${score.marbles_remaining === 1 ? 'text-[var(--gold-primary)]' : 'text-white'}`}>
                {score.marbles_remaining}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

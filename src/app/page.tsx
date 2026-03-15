'use client';

import React, { useState } from 'react';
import GameBoard from '@/components/GameBoard';
import Leaderboard from '@/components/Leaderboard';
import { Playfair_Display } from 'next/font/google';

const playfair = Playfair_Display({ subsets: ['latin'] });

export default function Home() {
  const [finalScore, setFinalScore] = useState<number | null>(null);

  const handleGameEnd = (marblesRemaining: number) => {
    setFinalScore(marblesRemaining);
  };

  const handleScoreSubmitted = () => {
    // Optionally reset finalScore if you want to let them immediately play again without seeing form
    // setFinalScore(null); 
  };

  return (
    <main className="min-h-screen bg-[var(--background)] text-[var(--foreground)] py-12 px-4 selection:bg-[var(--gold-primary)] selection:text-black">
      <div className="max-w-4xl mx-auto">
        
        <header className="text-center mb-10">
          <h1 className={`${playfair.className} text-5xl md:text-6xl text-transparent bg-clip-text bg-gradient-to-br from-[var(--gold-primary)] to-[#fffdf5] font-bold mb-4 tracking-tight drop-shadow-sm`}>
            Aurelian Solitaire
          </h1>
          <p className="text-gray-400 font-light tracking-wide max-w-xl mx-auto text-sm md:text-base">
            Jump pegs horizontally or vertically to clear the board. 
            Leave exactly one peg in the center to achieve perfection.
          </p>
        </header>

        <section className="flex flex-col lg:flex-row gap-12 items-start justify-center">
          <div className="flex-1 w-full">
            <GameBoard onGameEnd={handleGameEnd} />
          </div>
          
          <div className="w-full lg:w-1/3">
            <Leaderboard 
              currentScore={finalScore} 
              onScoreSubmitted={handleScoreSubmitted} 
            />
          </div>
        </section>

      </div>
    </main>
  );
}

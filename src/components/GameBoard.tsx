'use client';

import React, { useState, useEffect } from 'react';
import confetti from 'canvas-confetti';

type Position = { r: number; c: number };
type CellType = 'O' | '.' | ' ';
type BoardState = CellType[][];

const INITIAL_BOARD: BoardState = [
  [' ', ' ', 'O', 'O', 'O', ' ', ' '],
  [' ', ' ', 'O', 'O', 'O', ' ', ' '],
  ['O', 'O', 'O', 'O', 'O', 'O', 'O'],
  ['O', 'O', 'O', '.', 'O', 'O', 'O'],
  ['O', 'O', 'O', 'O', 'O', 'O', 'O'],
  [' ', ' ', 'O', 'O', 'O', ' ', ' '],
  [' ', ' ', 'O', 'O', 'O', ' ', ' ']
];

interface GameBoardProps {
  onGameEnd: (marblesRemaining: number) => void;
}

export default function GameBoard({ onGameEnd }: GameBoardProps) {
  const [board, setBoard] = useState<BoardState>(INITIAL_BOARD);
  const [selected, setSelected] = useState<Position | null>(null);
  const [marblesRemaining, setMarblesRemaining] = useState(32);
  const [gameOver, setGameOver] = useState(false);
  const [win, setWin] = useState(false);
  const [validMoves, setValidMoves] = useState<Position[]>([]);

  // Calculate available moves for a specific pos
  const getValidMovesForMarble = (r: number, c: number, currentBoard: BoardState): Position[] => {
    const moves: Position[] = [];
    const directions = [[0, 2], [0, -2], [2, 0], [-2, 0]]; // Right, Left, Down, Up

    directions.forEach(([dr, dc]) => {
      const nr = r + dr;
      const nc = c + dc;
      const midR = r + dr / 2;
      const midC = c + dc / 2;

      if (
        nr >= 0 && nr < 7 &&
        nc >= 0 && nc < 7 &&
        currentBoard[nr][nc] === '.' &&
        currentBoard[midR][midC] === 'O'
      ) {
        moves.push({ r: nr, c: nc });
      }
    });
    return moves;
  };

  // Check if game is over (no valid moves left)
  const checkGameOver = (currentBoard: BoardState) => {
    let marblesCount = 0;
    let hasMoves = false;

    for (let r = 0; r < 7; r++) {
      for (let c = 0; c < 7; c++) {
        if (currentBoard[r][c] === 'O') {
          marblesCount++;
          if (getValidMovesForMarble(r, c, currentBoard).length > 0) {
            hasMoves = true;
          }
        }
      }
    }

    setMarblesRemaining(marblesCount);

    if (!hasMoves) {
      setGameOver(true);
      // Win condition: 1 marble exactly in the center (3,3)
      if (marblesCount === 1 && currentBoard[3][3] === 'O') {
        setWin(true);
        triggerConfetti();
      }
      onGameEnd(marblesCount);
    }
  };

  const triggerConfetti = () => {
    const duration = 3000;
    const end = Date.now() + duration;

    const frame = () => {
      confetti({
        particleCount: 5,
        angle: 60,
        spread: 55,
        origin: { x: 0 },
        colors: ['#d4af37', '#ffd700', '#ffffff'] // Gold theme confetti
      });
      confetti({
        particleCount: 5,
        angle: 120,
        spread: 55,
        origin: { x: 1 },
        colors: ['#d4af37', '#ffd700', '#ffffff']
      });

      if (Date.now() < end) {
        requestAnimationFrame(frame);
      }
    };
    frame();
  };

  const handleClick = (r: number, c: number) => {
    if (gameOver) return;

    const cellVal = board[r][c];

    if (cellVal === ' ') return; // Blocked area

    if (!selected) {
      if (cellVal === 'O') {
        setSelected({ r, c });
        setValidMoves(getValidMovesForMarble(r, c, board));
      }
    } else {
      // Trying to move
      const isValidMove = validMoves.some(m => m.r === r && m.c === c);

      if (isValidMove) {
        // Execute move
        const newBoard = board.map(row => [...row]);
        const midR = selected.r + (r - selected.r) / 2;
        const midC = selected.c + (c - selected.c) / 2;

        newBoard[selected.r][selected.c] = '.';
        newBoard[midR][midC] = '.';
        newBoard[r][c] = 'O';

        setBoard(newBoard);
        setSelected(null);
        setValidMoves([]);
        checkGameOver(newBoard); // Check after move
      } else {
        // Did not click a valid move destination.
        // If clicked another marble, select it instead.
        if (cellVal === 'O') {
          setSelected({ r, c });
          setValidMoves(getValidMovesForMarble(r, c, board));
        } else {
          // Clicked empty space or same marble, deselect
          setSelected(null);
          setValidMoves([]);
        }
      }
    }
  };

  const resetGame = () => {
    setBoard(INITIAL_BOARD);
    setSelected(null);
    setValidMoves([]);
    setGameOver(false);
    setWin(false);
    setMarblesRemaining(32);
  };

  return (
    <div className="flex flex-col items-center justify-center p-8 w-full max-w-2xl mx-auto">
      <div className="mb-8 text-center flex justify-between w-full items-end">
        <div>
          <h2 className="text-xl font-light text-[var(--gold-secondary)] uppercase tracking-widest mb-1">Status</h2>
          <div className="text-3xl font-semibold">
            {gameOver ? (win ? <span className="text-[var(--gold-primary)] font-bold">VICTORY</span> : 'GAME OVER') : 'PLAYING'}
          </div>
        </div>
        <div className="text-right">
          <h2 className="text-sm font-light text-gray-400 uppercase tracking-widest mb-1">Remaining</h2>
          <div className="text-4xl font-bold font-mono text-[var(--gold-primary)]">{marblesRemaining}</div>
        </div>
      </div>

      <div className="inline-block p-4 bg-[var(--cell-bg)] rounded-xl border border-[var(--cell-border)] shadow-2xl relative">
        {board.map((row, r) => (
          <div key={r} className="flex">
            {row.map((cell, c) => {
              const isBlocked = cell === ' ';
              const isMarble = cell === 'O';
              const isSelected = selected && selected.r === r && selected.c === c;
              const isValidMove = validMoves.some(m => m.r === r && m.c === c);

              return (
                <div
                  key={`${r}-${c}`}
                  onClick={() => handleClick(r, c)}
                  className={`
                    w-10 h-10 sm:w-14 sm:h-14 m-1 sm:m-1.5 rounded-full flex items-center justify-center
                    transition-all duration-300
                    ${isBlocked ? 'invisible' : 'bg-[#1a1a1a] shadow-inner cursor-pointer'}
                    ${!isBlocked && !isMarble ? 'hover:bg-[#2a2a2a]' : ''}
                    ${isValidMove ? 'ring-2 ring-[var(--gold-primary)] bg-[#2a2a2a]' : ''}
                  `}
                >
                  {isMarble && (
                    <div
                      className={`
                        w-8 h-8 sm:w-11 sm:h-11 rounded-full marble-enter
                        ${isSelected ? 'selected-glow' : ''}
                      `}
                      style={{
                        background: isSelected ? 'var(--marble-selected)' : 'var(--marble-bg)',
                        boxShadow: 'var(--marble-shadow)'
                      }}
                    />
                  )}
                  {isValidMove && !isMarble && (
                   <div className="w-3 h-3 rounded-full bg-[var(--gold-primary)] opacity-50" />
                  )}
                </div>
              );
            })}
          </div>
        ))}
        {gameOver && (
          <div className="absolute inset-0 bg-black/70 rounded-xl flex items-center justify-center flex-col animate-in fade-in duration-500 backdrop-blur-sm z-10">
             <div className={`text-4xl font-bold mb-4 ${win ? 'text-[var(--gold-primary)]' : 'text-gray-300'}`}>
               {win ? 'Perfect Score!' : 'No More Moves'}
             </div>
             <p className="text-lg mb-6 text-gray-400">Marbles left: {marblesRemaining}</p>
             <button
               onClick={resetGame}
               className="px-6 py-3 bg-transparent border border-[var(--gold-primary)] text-[var(--gold-primary)] hover:bg-[var(--gold-primary)] hover:text-black transition-colors rounded uppercase tracking-widest font-semibold"
             >
               Play Again
             </button>
          </div>
        )}
      </div>
    </div>
  );
}

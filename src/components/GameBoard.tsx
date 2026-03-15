'use client';

import React, { useState, useEffect } from 'react';
import confetti from 'canvas-confetti';
import { saveTrajectoryAction, TrajectoryEntry } from '@/app/actions';
import { getHintFromONNX } from '@/lib/onnxInference';

type Position = { r: number; c: number };
type CellType = 'O' | '.' | ' ';
type BoardState = CellType[][];
type Move = { r1: number; c1: number; r2: number; c2: number };

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
  
  // Hint State
  const [hintLoading, setHintLoading] = useState(false);
  const [hintMove, setHintMove] = useState<Move | null>(null);
  const [hintMessage, setHintMessage] = useState<string | null>(null);
  
  // RL Data Collection
  const [trajectory, setTrajectory] = useState<TrajectoryEntry['moves']>([]);

  // Clear hint when user interacts
  useEffect(() => {
    setHintMove(null);
    setHintMessage(null);
  }, [board, selected]);

  const handleGetHint = async () => {
    setHintLoading(true);
    setHintMessage(null);
    setHintMove(null);

    try {
      // 1. Convert board to compact string format
      let boardStr = "";
      for (let r=0; r<7; r++) {
          for (let c=0; c<7; c++) {
              boardStr += board[r][c];
          }
      }

      // 2. Call Local Browser ML Inference (Zero-Cost, Instantaneous!)
      const startTime = performance.now();
      const action = await getHintFromONNX(boardStr);
      const endTime = performance.now();

      if (action) {
        // action is {r, c, d} where d is 0=Up, 1=Right, 2=Down, 3=Left
        const DIRS = [[-1, 0], [0, 1], [1, 0], [0, -1]];
        const dr = DIRS[action.d][0];
        const dc = DIRS[action.d][1];
        
        const move: Move = {
            r1: action.r,
            c1: action.c,
            r2: action.r + 2*dr,
            c2: action.c + 2*dc
        };
        
        setHintMove(move);
        const ms = (endTime - startTime).toFixed(0);
        setHintMessage(`Move the blue peg to the blue target! (${ms}ms)`);
      } else {
        setHintMessage("ML Model couldn't find a valid move.");
      }
    } catch (error) {
      console.error("Failed to get ML hint:", error);
      setHintMessage("Failed to load Neural Network.");
    } finally {
      setHintLoading(false);
    }
  };

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
  const checkGameOver = async (currentBoard: BoardState) => {
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
      
      // Submit RL Data in the background
      saveTrajectoryAction(marblesCount, trajectory).catch(err => {
        console.error("Failed to silently upload trajectory", err);
      });
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

        // RL Formatting: Determine Direction (0: Up, 1: Right, 2: Down, 3: Left)
        let dir = -1;
        if (r < selected.r) dir = 0; // Up
        else if (c > selected.c) dir = 1; // Right
        else if (r > selected.r) dir = 2; // Down
        else if (c < selected.c) dir = 3; // Left

        // Convert board to compact string
        let stateStr = "";
        for (let br=0; br<7; br++) {
            for (let bc=0; bc<7; bc++) {
                stateStr += board[br][bc];
            }
        }

        const moveRecord = {
            state: stateStr,
            action: { row: selected.r, col: selected.c, dir }
        };

        setTrajectory(prev => [...prev, moveRecord]);
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
    setHintMove(null);
    setHintMessage(null);
    setTrajectory([]);
  };

  return (
    <div className="flex flex-col items-center justify-center p-8 w-full max-w-2xl mx-auto">
      <div className="mb-4 text-center flex justify-between w-full items-end">
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
      
      {/* Hint Controls */}
      <div className="w-full flex justify-between items-center mb-6 h-10">
            <button
                onClick={handleGetHint}
                disabled={gameOver || hintLoading}
                className={`flex items-center space-x-2 px-6 py-2 rounded-full border border-[--gold-primary] text-[--gold-primary] transition-all duration-300 shadow-[0_0_10px_rgba(212,175,55,0.1)] 
                           ${gameOver ? 'opacity-50 cursor-not-allowed' : 'hover:bg-[--gold-primary] hover:text-[#111111] hover:shadow-[0_0_20px_rgba(212,175,55,0.4)]'}`}
            >
                <span>{hintLoading ? '🧠 Thinking...' : '💡 Get AI Hint'}</span>
            </button>
        {hintMessage && (
          <div className="text-xs text-amber-500 animate-in fade-in max-w-[200px] text-right">
            {hintMessage}
          </div>
        )}
      </div>

      <div className="inline-block p-4 bg-[var(--cell-bg)] rounded-xl border border-[var(--cell-border)] shadow-2xl relative">
        {board.map((row, r) => (
          <div key={r} className="flex">
            {row.map((cell, c) => {
              const isBlocked = cell === ' ';
              const isMarble = cell === 'O';
              const isSelected = selected && selected.r === r && selected.c === c;
              const isValidMove = validMoves.some(m => m.r === r && m.c === c);
              
              // Hint highlighting
              const isHintOrigin = hintMove && hintMove.r1 === r && hintMove.c1 === c;
              const isHintDest = hintMove && hintMove.r2 === r && hintMove.c2 === c;

              return (
                <div
                  key={`${r}-${c}`}
                  onClick={() => handleClick(r, c)}
                  className={`
                    w-10 h-10 sm:w-14 sm:h-14 m-1 sm:m-1.5 rounded-full flex items-center justify-center
                    transition-all duration-300 relative
                    ${isBlocked ? 'invisible' : 'bg-[#1a1a1a] shadow-inner cursor-pointer'}
                    ${!isBlocked && !isMarble ? 'hover:bg-[#2a2a2a]' : ''}
                    ${isValidMove && !isHintDest ? 'ring-2 ring-[var(--gold-primary)] bg-[#2a2a2a]' : ''}
                    ${isHintDest ? 'ring-2 ring-blue-500 bg-blue-500/10 shadow-[0_0_15px_rgba(59,130,246,0.5)]' : ''}
                  `}
                >
                  {isMarble && (
                    <div
                      className={`
                        w-8 h-8 sm:w-11 sm:h-11 rounded-full marble-enter
                        ${isSelected && !isHintOrigin ? 'selected-glow' : ''}
                        ${isHintOrigin ? 'ring-4 ring-blue-500 ring-offset-2 ring-offset-[#1a1a1a] shadow-[0_0_20px_rgba(59,130,246,0.8)] animate-pulse' : ''}
                      `}
                      style={{
                        background: isSelected ? 'var(--marble-selected)' : 'var(--marble-bg)',
                        boxShadow: 'var(--marble-shadow)'
                      }}
                    />
                  )}
                  {(isValidMove || isHintDest) && !isMarble && (
                   <div className={`w-3 h-3 rounded-full ${isHintDest ? 'w-4 h-4 bg-blue-500 animate-ping opacity-100 ring-2 ring-blue-300' : 'bg-[var(--gold-primary)] opacity-50'}`} />
                  )}
                </div>
              );
            })}
          </div>
        ))}
        {gameOver && (
          <div className="absolute inset-0 bg-black/80 rounded-xl flex items-center justify-center flex-col animate-in fade-in duration-500 backdrop-blur-sm z-10">
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

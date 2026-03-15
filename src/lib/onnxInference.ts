import * as ort from 'onnxruntime-web';

// Initialize ONNX runtime configuration
ort.env.wasm.numThreads = 1;

export async function getHintFromONNX(boardStr: string): Promise<{r: number, c: number, d: number} | null> {
    try {
        // Load the model from the public directory
        // This downloads the ~7MB file to the browser cache on first use
        const session = await ort.InferenceSession.create('/peg_solitaire_policy.onnx');

        // Convert board string ('O', '.', ' ') to a Float32Array matching the PyTorch tensor setup
        const mapping: Record<string, number> = {' ': -1, '.': 0, 'O': 1};
        const inputMap = new Float32Array(49);
        for (let i = 0; i < 49; i++) {
            inputMap[i] = mapping[boardStr[i]] ?? -1;
        }

        // Create the tensor: float32, batch_size=1, channel=1, height=7, width=7
        const tensor = new ort.Tensor('float32', inputMap, [1, 1, 7, 7]);
        const feeds = { 'board_state': tensor };

        // Run inference!
        const results = await session.run(feeds);
        
        // This is the output of the final Linear layer (196 logits)
        const logits = results.action_logits.data as Float32Array;
        
        // Determine all legal moves (we must mask out illegal moves just like we did in evaluate.py)
        const legalMoves = getLegalMoves(boardStr);
        if (legalMoves.length === 0) return null;

        // Apply masking: set logits of illegal moves to negative infinity
        const maskedLogits = new Float32Array(196);
        maskedLogits.fill(-Infinity);
        
        for (const move of legalMoves) {
            const { r, c, d } = move;
            const actionIdx = (r * 7 * 4) + (c * 4) + d;
            maskedLogits[actionIdx] = logits[actionIdx];
        }

        // Find the argmax among the masked logits
        let bestIdx = -1;
        let bestVal = -Infinity;
        for (let i = 0; i < 196; i++) {
            if (maskedLogits[i] > bestVal) {
                bestVal = maskedLogits[i];
                bestIdx = i;
            }
        }

        if (bestIdx === -1) return null;

        // Decode the index back to row, col, direction
        const d = bestIdx % 4;
        const c = Math.floor(bestIdx / 4) % 7;
        const r = Math.floor(bestIdx / 28) % 7;

        return { r, c, d };
        
    } catch (e) {
        console.error("ONNX Inference failed:", e);
        return null;
    }
}

// Helper to calculate exact legal moves so the Neural Network doesn't break the rules
function getLegalMoves(boardStr: string) {
    const moves: {r: number, c: number, d: number}[] = [];
    const DIRS = [[-1, 0], [0, 1], [1, 0], [0, -1]]; // 0: Up, 1: Right, 2: Down, 3: Left
    
    // boardStr is 49 chars flat
    const getCell = (r: number, c: number) => {
        if (r < 0 || r >= 7 || c < 0 || c >= 7) return ' ';
        return boardStr[r * 7 + c];
    };

    for (let r = 0; r < 7; r++) {
        for (let c = 0; c < 7; c++) {
            if (getCell(r, c) === 'O') {
                for (let d = 0; d < 4; d++) {
                    const dr = DIRS[d][0];
                    const dc = DIRS[d][1];
                    const r1 = r + dr, c1 = c + dc;
                    const r2 = r + 2*dr, c2 = c + 2*dc;
                    
                    if (getCell(r1, c1) === 'O' && getCell(r2, c2) === '.') {
                        moves.push({ r, c, d });
                    }
                }
            }
        }
    }
    return moves;
}

/** A spread point in the books' sign: "−3.5" (favoured by 3.5), "+3.5" (getting 3.5), "PK" for 0. */
export const signedLine = (n: number): string => (n === 0 ? "PK" : `${n > 0 ? "+" : "−"}${Math.abs(n).toFixed(1)}`);

import { useId } from "react";

/**
 * StratoSense mark (v0 foundation): an Earth disc built from pixel cells, a few of them lit as a
 * cloud mask, crossed by an orbital ring with a small satellite. Final brand assets come later.
 */
export function LogoMark({ size = 32, title }: { size?: number; title?: string }) {
  const clipId = useId();
  const cell = 4.5;
  const x0 = 10.5;
  // Grid cells (col,row) lit as the "cloud".
  const cloud: Array<[number, number]> = [
    [1, 1], [2, 1], [3, 1],
    [0, 2], [1, 2], [2, 2], [3, 2], [4, 2],
    [1, 3], [2, 3],
  ];
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 48 48"
      role={title ? "img" : undefined}
      aria-label={title}
      aria-hidden={title ? undefined : true}
    >
      <defs>
        <clipPath id={clipId}>
          <circle cx="24" cy="24" r="12" />
        </clipPath>
      </defs>
      <g clipPath={`url(#${clipId})`}>
        <rect width="48" height="48" fill="var(--accent)" fillOpacity="0.2" />
        {cloud.map(([c, r]) => (
          <rect
            key={`${c}-${r}`}
            x={x0 + c * cell + 0.4}
            y={x0 + r * cell + 0.4}
            width={cell - 0.8}
            height={cell - 0.8}
            fill="var(--text)"
          />
        ))}
      </g>
      <circle cx="24" cy="24" r="12" fill="none" stroke="var(--accent)" strokeWidth="1.5" />
      <g transform="rotate(-28 24 24)">
        <ellipse cx="24" cy="24" rx="20" ry="7" fill="none" stroke="var(--accent-strong)" strokeWidth="1.4" />
        <rect x="42" y="22" width="4" height="4" fill="var(--warn)" />
      </g>
    </svg>
  );
}

export function Logo({ size = 32 }: { size?: number }) {
  return (
    <span className="inline-flex items-center gap-2.5">
      <LogoMark size={size} />
      <span className="text-lg font-semibold tracking-tight">StratoSense</span>
    </span>
  );
}

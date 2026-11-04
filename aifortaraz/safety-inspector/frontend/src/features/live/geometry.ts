export interface FrameRect {
  x: number;
  y: number;
  width: number;
  height: number;
}
export function containRect(
  frameWidth: number,
  frameHeight: number,
  width: number,
  height: number,
): FrameRect {
  if (frameWidth <= 0 || frameHeight <= 0 || width <= 0 || height <= 0)
    return { x: 0, y: 0, width: 0, height: 0 };
  const scale = Math.min(width / frameWidth, height / frameHeight);
  const fittedWidth = frameWidth * scale;
  const fittedHeight = frameHeight * scale;
  return {
    x: (width - fittedWidth) / 2,
    y: (height - fittedHeight) / 2,
    width: fittedWidth,
    height: fittedHeight,
  };
}
export function projectBox(
  box: readonly [number, number, number, number],
  rect: FrameRect,
): FrameRect {
  return {
    x: rect.x + box[0] * rect.width,
    y: rect.y + box[1] * rect.height,
    width: box[2] * rect.width,
    height: box[3] * rect.height,
  };
}
export function detectionOpacity(ageMs: number) {
  return ageMs <= 1000 ? 1 : ageMs >= 3000 ? 0 : 1 - (ageMs - 1000) / 2000;
}

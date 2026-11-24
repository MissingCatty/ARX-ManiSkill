export function preferredEncodings(header = '') {
  const qualities = new Map();
  for (const part of header.toLowerCase().split(',')) {
    const [coding, ...parameters] = part.trim().split(';');
    const quality = parameters.map(item => item.trim()).find(item => item.startsWith('q='));
    const value = quality ? Number(quality.slice(2)) : 1;
    qualities.set(coding.trim(), Number.isFinite(value) && value >= 0 && value <= 1 ? value : 0);
  }
  const quality = coding => qualities.get(coding) ?? qualities.get('*') ?? 0;
  const identity = qualities.get('identity') ?? 0;
  return ['br', 'gzip'].filter(coding => quality(coding) > 0 && quality(coding) >= identity)
    .sort((a, b) => quality(b) - quality(a));
}

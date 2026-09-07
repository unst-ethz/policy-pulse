import { describe, expect, it } from 'vitest';
import { agreementColor } from './mapColors';

describe('agreement map interpretation', () => {
  it('retains the original opposite, neutral and identical vote colours', () => {
    expect(agreementColor(0)).toBe('rgb(165, 0, 38)');
    expect(agreementColor(0.5)).toBe('rgb(255, 255, 191)');
    expect(agreementColor(1)).toBe('rgb(49, 54, 149)');
  });
  it('keeps missing observations grey and separate from a measured zero', () => {
    expect(agreementColor(null)).toBe('#e1e1e1');
    expect(agreementColor(undefined)).toBe('#e1e1e1');
    expect(agreementColor(0)).not.toBe(agreementColor(null));
  });
  it('moves yellow to the consensus midpoint while retaining both endpoints', () => {
    expect(agreementColor(0.8, 0.8)).toBe(agreementColor(0.5));
    expect(agreementColor(0, 0.8)).toBe(agreementColor(0));
    expect(agreementColor(1, 0.8)).toBe(agreementColor(1));
  });
});

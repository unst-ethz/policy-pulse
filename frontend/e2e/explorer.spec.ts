import { expect, test } from '@playwright/test';

test('browse, paginate, filter and download the same records', async ({ page }) => {
  await page.goto('/trends');
  await expect(page.getByText('160 matching records')).toBeVisible();
  await page.getByRole('button', { name: 'Next', exact: true }).click();
  await expect(page.getByText('21–40 of 160')).toBeVisible();
  await page.getByLabel('Reference country').selectOption('USA');
  await page.getByLabel('Keyword expression').fill('nuclear & disarmament');
  await page.getByRole('button', { name: 'Apply filters' }).click();
  await expect(page.getByText('80 matching records')).toBeVisible();
  await expect(page).toHaveURL(/country=USA/);
  await page.reload();
  await expect(page.getByLabel('Keyword expression')).toHaveValue('nuclear & disarmament');
  const download = page.waitForEvent('download');
  await page.getByRole('link', { name: 'Download CSV' }).click();
  expect((await download).suggestedFilename()).toBe('policy-pulse-resolutions.csv');
  await page.getByLabel('Recorded vote').selectOption('N');
  await expect(
    page.locator('.resolution-card').first().getByText('No', { exact: true }),
  ).toBeVisible();
});

test('date validation and empty selections remain explicit', async ({ page }) => {
  await page.goto('/trends');
  await page.getByLabel('From', { exact: true }).fill('2025-01-01');
  await page.getByLabel('To', { exact: true }).fill('2024-01-01');
  await page.getByRole('button', { name: 'Apply filters' }).click();
  await expect(page.getByRole('alert')).toHaveText('Start date must be on or before end date.');
  await page.getByLabel('From', { exact: true }).fill('1800-01-01');
  await page.getByLabel('To', { exact: true }).fill('1800-12-31');
  await page.getByRole('button', { name: 'Apply filters' }).click();
  await expect(page.getByText('0 matching records')).toBeVisible();
  await page.getByRole('button', { name: 'Multilateral alignment', exact: true }).click();
  await expect(
    page.getByRole('heading', { name: 'No countries meet the sample threshold' }),
  ).toBeVisible();
});

test('map, timeline, subjects and multilateral views use the API', async ({ page }, testInfo) => {
  await page.goto('/trends?country=USA&compare=CHE&view=map');
  await expect(
    page.getByRole('img', { name: 'World map of bilateral voting agreement' }),
  ).toBeVisible();
  await expect(page.getByText('this is not an official UN map', { exact: false })).toBeVisible();
  await page.getByLabel('Centre colour scale on average consensus score').check();
  await expect(page.getByText('Yellow midpoint:', { exact: false })).toBeVisible();
  await page.getByLabel('Centre colour scale on average consensus score').uncheck();
  await page.screenshot({ path: testInfo.outputPath('agreement-map.png'), fullPage: true });
  await page.getByRole('button', { name: 'Agreement timeline', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Agreement over time' })).toBeVisible();
  await expect(page.locator('.recharts-line')).not.toHaveCount(0);
  await page.getByLabel('Include special and emergency sessions').check();
  await page.getByText('View session values and sample sizes').click();
  await expect(page.getByRole('cell', { name: '1sp', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'By subject', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Disarmament →' })).toBeVisible();
  await page.getByRole('button', { name: 'Disarmament →' }).click();
  await expect(
    page.getByRole('button', { name: 'Nuclear disarmament', exact: true }),
  ).toBeVisible();
  await page.getByRole('button', { name: 'Multilateral alignment', exact: true }).click();
  await expect(page.getByLabel('Vertical axis')).toBeVisible();
  await page.getByLabel('Vertical axis').selectOption('yes_rate');
  await page.getByText('View country statistics').click();
  await expect(page.getByRole('columnheader', { name: 'Votes cast', exact: true })).toBeVisible();
});

test('homepage retains the Policy Pulse identity and links into the explorer', async ({
  page,
}, testInfo) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Welcome to Policy Pulse' })).toBeVisible();
  await expect(page.getByRole('link', { name: 'UN-ETH Policy Pulse', exact: true })).toBeVisible();
  await expect(page.locator('.hero-image')).toBeVisible();
  await expect(page.locator('.recent-grid .resolution-card')).toHaveCount(6);
  await page.screenshot({ path: testInfo.outputPath('homepage-desktop.png'), fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
  ).toBeTruthy();
  await page.screenshot({ path: testInfo.outputPath('homepage-mobile.png'), fullPage: true });
  await page.getByRole('link', { name: 'Explore Data →', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Explore GA Votes Yourself' })).toBeVisible();
});

test('subject word cloud drills into matching resolutions', async ({ page }) => {
  await page.goto('/trends?view=words');
  await page.getByLabel('Term source').selectOption('category');
  await page.getByRole('button', { name: 'nuclear disarmament', exact: true }).click();
  await expect(page).toHaveURL(/view=resolutions/);
  await expect(page).toHaveURL(/subject=/);
  await expect(page.locator('.resolution-card')).not.toHaveCount(0);
});

test('resolution details distinguish non-votes and missing values', async ({ page }) => {
  await page.goto('/resolutions/9000000');
  await expect(page.getByRole('heading', { name: /Synthetic test resolution 1:/ })).toBeVisible();
  await page.getByLabel('Search country votes').fill('Germany');
  await expect(page.getByText('Non-member / no data', { exact: true })).toBeVisible();
  await expect(page.getByRole('link', { name: 'Read the original UN record' })).toHaveAttribute(
    'href',
    'https://digitallibrary.un.org/record/9000000',
  );
});

test('legacy profile links work and printing is available', async ({ page }) => {
  await page.goto('/profile?country1=USA&country2=CHE');
  await expect(
    page.getByRole('heading', { name: 'United States of America', exact: true }),
  ).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Most aligned voting records' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Print / Save as PDF' })).toBeVisible();
});

test('API failure is never displayed as an empty result', async ({ page }) => {
  await page.route('**/api/v1/resolutions?**', (route) =>
    route.fulfill({
      status: 503,
      contentType: 'application/json',
      body: JSON.stringify({ detail: { message: 'The dataset is loading or unavailable.' } }),
    }),
  );
  await page.goto('/trends');
  await expect(page.getByRole('alert')).toContainText('Dataset temporarily unavailable', {
    timeout: 15_000,
  });
  await expect(page.getByText('0 matching records')).not.toBeVisible();
});

test('mobile layout fits and the methodology is available', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/trends?country=USA');
  await expect(page.locator('.resolution-card').first()).toBeVisible();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
  ).toBeTruthy();
  await page.getByRole('link', { name: 'Methodology', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'The agreement formula' })).toBeVisible();
});

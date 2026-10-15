/**
 * On-demand TestDriver check of the Rate vs Poisson still in this checkout.
 *
 * TD_API_KEY is required. From the repository root:
 *   TD_API_KEY=... npx vitest run tests/hero.test.js
 *
 * pytest and CI do not collect this file.
 */
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { describe, expect, it } from "vitest";
import { TestDriver } from "testdriverai/vitest/hooks";

const heroStillUrl = pathToFileURL(
  path.resolve(
    path.dirname(fileURLToPath(import.meta.url)),
    "../docs/images/rate-vs-poisson.png",
  ),
).href;

describe.skipIf(!process.env.TD_API_KEY)("Rate vs Poisson Hero Still", () => {
  it("should display the hero still image with both panels and complete provenance", async (context) => {
    const testdriver = TestDriver(context);

    await testdriver.provision.chrome({
      url: heroStillUrl,
      maximized: true,
    });

    const rasterAssert = await testdriver.assert(
      "Rate vs Poisson hero raster image is displayed with both RateEncoder (deterministic streaming) and PoissonEncoder (stochastic) panels visible",
    );
    expect(rasterAssert).toBeTruthy();

    const metadataAssert = await testdriver.assert(
      "The image caption displays seed=1592590337, dt_seconds=0.001, N=8, T=64 and commit 908a34797c8aaff0a57e0ea4167aa00f3780c940",
    );
    expect(metadataAssert).toBeTruthy();
  });
});

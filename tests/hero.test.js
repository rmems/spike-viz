/**
 * On-demand TestDriver check of the checked-in Rate vs Poisson still.
 *
 * pytest and CI do not collect this file. Vitest skips it unless TD_API_KEY
 * is set. Chrome opens the PNG already on main (the #48 still); it does not
 * use a tunnel and it does not encode or invent spikes.
 */
import { describe, expect, it } from "vitest";
import { TestDriver } from "testdriverai/vitest/hooks";

// Same blob as origin/main docs/images/rate-vs-poisson.png (git 499ca36b).
const HERO_STILL_URL =
  "https://raw.githubusercontent.com/rmems/spike-viz/de73abb2ce89b6e6f235948fb582b901b99a8116/docs/images/rate-vs-poisson.png";

describe.skipIf(!process.env.TD_API_KEY)("Rate vs Poisson Hero Still", () => {
  it("should display the hero still image with both panels and complete provenance", async (context) => {
    const testdriver = TestDriver(context);

    await testdriver.provision.chrome({
      url: HERO_STILL_URL,
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

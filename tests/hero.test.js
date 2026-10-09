import { describe, expect, it } from "vitest";
import { TestDriver } from "testdriverai/vitest/hooks";

describe("Rate vs Poisson Hero Still", () => {
  it("should display the hero still image with both panels and complete provenance", async (context) => {
    const testdriver = TestDriver(context);

    await testdriver.provision.chrome({
      url: "https://new-bats-raise.loca.lt/docs/images/rate-vs-poisson.png",
      maximized: true,
    });

    // Handle localtunnel interstitial if present
    try {
      const ipInput = await testdriver.find("IP address input field", { timeout: 5000 });
      if (ipInput) {
        await ipInput.click();
        await testdriver.type("3.81.233.114");
        const continueBtn = await testdriver.find("Continue button or Click to Submit", { timeout: 5000 });
        await continueBtn.click();
      }
    } catch {
      // Direct navigation succeeded without interstitial
    }

    const rasterAssert = await testdriver.assert(
      "Rate vs Poisson hero raster image is displayed with both RateEncoder and PoissonEncoder panels visible"
    );
    expect(rasterAssert).toBeTruthy();

    const metadataAssert = await testdriver.assert(
      "The image caption displays seed=1592590337, dt_seconds=0.001, N=8, T=64 and commit becb40d0c8722710677dabbf66d420a9c77f2eda"
    );
    expect(metadataAssert).toBeTruthy();
  });
});

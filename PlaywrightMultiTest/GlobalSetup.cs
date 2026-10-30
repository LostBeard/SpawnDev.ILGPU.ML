namespace PlaywrightMultiTest
{
    [SetUpFixture]
    public class GlobalSetup
    {
        [OneTimeSetUp]
        public void SetUp()
        {
            // This tells Playwright to run in headed mode for the entire test run
            //Environment.SetEnvironmentVariable("HEADED", "1");
            // The fake camera / microphone flags are Chromium launch arguments (ProjectRunner.BuildChromiumArgs) plus the context's camera / microphone grant.
            // They used to be set here as a PLAYWRIGHT_ARGS environment variable, which Playwright for .NET does not read: no browser test could open a camera.
        }
    }
}

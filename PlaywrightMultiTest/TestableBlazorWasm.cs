using Microsoft.Playwright;
using PlaywrightMultiTest;

public class TestableBlazorWasm : TestableProject
{
    public StaticFileServer Server { get; set; }
    public IPlaywright Playwright { get; set; }
    public IBrowser Browser { get; set; }
    public IBrowserContext BrowserContext { get; set; }
    public IPage Page { get; set; }
    public string TestPage { get; set; } = "tests";
}

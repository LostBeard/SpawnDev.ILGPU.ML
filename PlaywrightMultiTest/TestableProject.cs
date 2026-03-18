using PlaywrightMultiTest;

public class TestableProject
{
    public ProjectDetails ProjectDetails { get; set; }
    public List<ProjectTest> Tests { get; set; } = new List<ProjectTest>();
}

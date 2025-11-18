// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-cv",
          title: "cv",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "post-evolving-code-at-test-time-building-a-mini-alphaevolve-on-my-laptop",
      
        title: "Evolving Code at Test Time — Building a Mini AlphaEvolve on My Laptop...",
      
      description: "Exploring how LLMs can write, test, and evolve code to solve problems, inspired by DeepMind&#39;s AlphaEvolve but built on a laptop.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2025/mini-evolve/";
        
      },
    },{id: "post-from-docs-to-answers-my-smolagents-docling-duckdb-experiment",
      
        title: "From Docs to Answers: My Smolagents + Docling + DuckDB Experiment 🤖",
      
      description: "Testing smolagents, docling, and duckdb through an agent that RAGs for you.",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2025/auto-rag-agent/";
        
      },
    },{id: "post-a-conditional-gan-for-data-augmentation-a-cautionary-tale",
      
        title: "A Conditional GAN for Data Augmentation: A Cautionary Tale",
      
      description: "Balancing an image classification dataset using synthetic images",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2025/cgan-class-balancer/";
        
      },
    },{id: "post-test-time-compute-my-take-on-story-generation",
      
        title: "Test-Time Compute: My Take on Story Generation",
      
      description: "Exploring test-time compute and beam search with a story generator app",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2025/test-time-compute-story-generation/";
        
      },
    },{id: "post-if-your-company-is-not-doing-ml-it-probably-won-t-start-any-time-soon-either",
      
        title: "If your company is not doing ML, it (probably) won’t start any time...",
      
      description: "",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/blog/2024/company-no-ml/";
        
      },
    },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/ssalb", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/ssalazaralbornoz", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];

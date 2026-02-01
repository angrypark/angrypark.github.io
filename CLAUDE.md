# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
npm run dev      # Start dev server at localhost:4321
npm run build    # TypeScript check + production build to ./dist/
npm run preview  # Preview production build locally
```

## Architecture

This is a personal tech blog built with Astro v5, Tailwind CSS v4, and MDX.

### Content System
- Blog posts: Markdown/MDX files in `src/content/blog/`
- Schema defined in `src/content/config.ts` - posts require `title`, `description`, `pubDate`
- Custom remark plugins in `src/plugins/`:
  - `remark-reading-time.mjs` - injects `minutesRead` into frontmatter
  - `remark-modified-time.mjs` - injects `lastModified` from git history

### Key Configuration
- `src/consts.ts` - Site title, description, social links, navigation links
- `astro.config.mjs` - Integrations: MDX, sitemap, astro-icon, partytown (for analytics)
- Math support via remark-math + rehype-katex

### Layouts
- `BaseLayout.astro` - Main layout with Navbar/Footer
- `BlogPostLayout.astro` - Blog post wrapper with TOC, comments (Giscus)

### Path Aliases (tsconfig.json)
- `@/*` → `src/*`
- `@blogimages/*` → `src/assets/blogimages/*`

### Theming
Dark/light mode via CSS custom properties and Tailwind classes (`dark:` prefix). Theme toggle in `ThemeSelector.astro`.

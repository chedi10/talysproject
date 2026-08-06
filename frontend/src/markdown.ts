/** Rendu Markdown léger sans dépendance externe */

export function renderSimpleMarkdown(text: string): string {
  let html = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
  html = html.replace(/^## (.+)$/gm, '<h3 class="mdH3">$1</h3>')
  html = html.replace(/^### (.+)$/gm, '<h4 class="mdH4">$1</h4>')
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
  html = html.replace(/`([^`]+)`/g, '<code class="mdCode">$1</code>')
  html = html.replace(/^\| .+\|$/gm, (line) => `<div class="mdTableRow">${line}</div>`)
  html = html.replace(/^- (.+)$/gm, '<li class="mdLi">$1</li>')
  html = html.replace(/(<li class="mdLi">[\s\S]*?<\/li>)+/g, (m) => `<ul class="mdUl">${m}</ul>`)
  html = html.replace(/\n\n/g, '</p><p class="mdP">')
  html = `<div class="mdBody"><p class="mdP">${html}</p></div>`
  return html
}

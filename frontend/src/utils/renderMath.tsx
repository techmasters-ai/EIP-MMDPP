import katex from "katex";
import "katex/dist/katex.min.css";

// Render inline ($...$) and display ($$...$$) LaTeX inside arbitrary text
// (TODO #73). Used by both the Docling viewer's image-description panel and
// the retrieval result cards' Full Text / preview. Conservative: only segments
// containing LaTeX-ish characters (\ ^ _ { }) are rendered as math, so prose
// like "$5 and $10" stays literal.
export function renderTextWithMath(text: string): Array<string | JSX.Element> {
  if (!text) return [text];
  const out: Array<string | JSX.Element> = [];
  const re = /\$\$([\s\S]+?)\$\$|\$([^$\n]+?)\$/g;
  let last = 0;
  let key = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    const latex = m[1] ?? m[2] ?? "";
    if (!/[\\^_{}]/.test(latex)) continue; // not math — leave the literal text
    if (m.index > last) out.push(text.slice(last, m.index));
    try {
      out.push(
        <span
          key={`tex-${key++}`}
          dangerouslySetInnerHTML={{
            __html: katex.renderToString(latex, {
              throwOnError: false,
              displayMode: m[1] !== undefined,
            }),
          }}
        />,
      );
    } catch {
      out.push(m[0]);
    }
    last = re.lastIndex;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

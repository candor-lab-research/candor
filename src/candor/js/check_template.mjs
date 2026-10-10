// Whether a chat template parses in @huggingface/jinja, the parser Bionic and LM Studio use.
// Usage: node check_template.mjs <chat_template.jinja> [template variables as JSON]
// Prints JSON: {"ok": true}, or {"ok": false, "error": ..., "lines": [line numbers of rejected statements]}.
import { Template } from "@huggingface/jinja";
import fs from "fs";

const [file, variables] = process.argv.slice(2);
const source = fs.readFileSync(file, "utf8");

try {
  new Template(source).render({
    messages: [{ role: "user", content: "hi" }],
    add_generation_prompt: true,
    bos_token: "",
    eos_token: "",
    ...(variables ? JSON.parse(variables) : {}),
  });
  console.log(JSON.stringify({ ok: true }));
} catch (error) {
  const closers = { if: "{% endif %}", for: "{% endfor %}", macro: "{% endmacro %}" };
  const lines = [];
  for (const match of source.matchAll(/\{%-?\s*([\s\S]*?)\s*-?%\}/g)) {
    const keyword = match[1].split(/\s+/)[0];
    if (!["if", "elif", "for", "set", "macro", "generation"].includes(keyword)) continue;
    const body = keyword === "elif" ? match[1].replace(/^elif/, "if") : match[1];
    let statement = `{% ${body} %}` + (closers[body.split(/\s+/)[0]] || "");
    if (keyword === "set" && !body.includes("=")) statement += "{% endset %}";
    try {
      new Template(statement);
    } catch {
      lines.push(source.slice(0, match.index).split("\n").length);
    }
  }
  console.log(JSON.stringify({ ok: false, error: error.message, lines }));
}

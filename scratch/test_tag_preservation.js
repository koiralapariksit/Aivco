const testHtmlSnippet = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>FC Barcelona</title>
    <style>
        body { background: #004d98; color: #a50044; }
    </style>
</head>
<body>
    <h1>FC Barcelona</h1>
    <p>Més que un club</p>
</body>
</html>`;

function cleanAndFormatTest(textWithCode) {
  let text = textWithCode;
  text = text.replace(/^(?:Claude responded:?\s*)?(?:Sharing files|Thinking|Thinking\.\.\.|Read a file, created a file, shared files|Thought for \d+ seconds?|Reviewing an attached task spec\.\.\.|View details)[\s\n]*/gi, '');
  text = text.replace(/(Reviewing an attached task spec\.\.\.?\s*)+/gi, '');
  text = text.replace(/(Thinking\.\.\.?\s*)+/gi, '');
  text = text.replace(/\bCopy\s*html\b/gi, '');
  text = text.replace(/\bCopy\s*code\b/gi, '');
  text = text.replace(/\b(?:just now|Copy|Retry)\b\s*$/gi, '');
  return text.trim();
}

console.log("Raw code has tags:", testHtmlSnippet.includes("<!DOCTYPE html>"));
console.log("Cleaned code has tags:", cleanAndFormatTest(testHtmlSnippet).includes("<!DOCTYPE html>"));


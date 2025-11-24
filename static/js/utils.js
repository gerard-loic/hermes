function jsonToText(jsonObject) {
  return Object.entries(jsonObject)
    .map(([label, value]) => `${label}:${value}`)
    .join('\n');
}

function textToJSON(text) {
  const lines = text.trim().split('\n');
  const result = {};
  
  lines.forEach(line => {
    const [label, ...valueParts] = line.split(':');
    if (label && valueParts.length > 0) {
      // On rejoint les parties au cas où la valeur contient des ":"
      result[label.trim()] = valueParts.join(':').trim();
    }
  });
  
  // Retourne une chaîne JSON avec des guillemets doubles
  return JSON.stringify(result);
}
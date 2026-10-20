// Scenarios for the text and audio demos (see _routes.mjs for the scenario contract).

export const scenarios = {
  '/sentiment': {
    run: async h => {
      await h.type('.sentiment-textarea', 'I absolutely love this movie, it was wonderful and moving!');
      await h.click('.run-btn');
      const verdict = await h.waitFor(`document.querySelector('.sentiment-verdict')?.innerText?.trim()`, 300000, '.sentiment-verdict');
      if (!/POSITIVE/i.test(verdict)) throw new Error('expected POSITIVE, the page said: ' + verdict);
      return `verdict: ${verdict}`;
    },
  },

  '/embeddings': {
    run: async h => {
      await h.typeNth('.sim-input', 0, 'The cat sat quietly on the warm mat.');
      await h.typeNth('.sim-input', 1, 'A kitten was resting on a soft rug.');
      await h.click('.run-btn');
      const score = await h.waitFor(`document.querySelector('.sim-score-value')?.innerText?.trim()`, 300000, '.sim-score-value');
      if (!/\d/.test(score)) throw new Error('no similarity score rendered: ' + score);
      return `similarity: ${score}`;
    },
  },
};

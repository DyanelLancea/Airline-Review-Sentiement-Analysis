document.addEventListener('DOMContentLoaded', () => {
  function buildChunked(box, sentences, chunkSize){
    let idx = 0;
    const textEl = document.createElement('span');

    const more = document.createElement('a');
    more.href = '#';
    more.className = 'more-link';

    function setLabel(){
      const left = Math.max(0, sentences.length - idx);
      if (left === 0) { more.remove(); return; }
      const step = Math.min(chunkSize, left);
      more.textContent = `… show ${step} more`;
    }

    function addChunk(e){
      if (e) e.preventDefault();
      const end = Math.min(idx + chunkSize, sentences.length);
      const part = sentences.slice(idx, end).join(' ');
      textEl.textContent += (idx ? ' ' : '') + part;
      idx = end;
      setLabel();
    }

    more.addEventListener('click', addChunk);
    more.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); addChunk(); }
    });

    box.appendChild(textEl);
    box.appendChild(document.createTextNode(' '));
    box.appendChild(more);

    setLabel();
    addChunk(); // show first chunk immediately
  }

  document.querySelectorAll('.chunked').forEach(box => {
    const sentences = JSON.parse(box.dataset.sentences || '[]');
    const chunkSize = parseInt(box.dataset.chunk || '3', 10);
    buildChunked(box, sentences, chunkSize);
  });
});
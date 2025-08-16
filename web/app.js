const API = 'http://localhost:8000/generate';
const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send');

function add(role, text) {
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  d.textContent = text;
  messagesEl.appendChild(d);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return d;
}

sendBtn.onclick = () => {
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = '';
  add('user', text);
  const prompt = `<|system|>You are a helpful assistant.<|user|>${text}`;
  const d = add('assistant', '');
  const url = API + '?prompt=' + encodeURIComponent(prompt);
  const es = new EventSource(url);
  sendBtn.disabled = true;
  es.onmessage = (e) => {
    d.textContent += e.data;
  };
  es.onerror = () => { es.close(); sendBtn.disabled = false; };
  es.addEventListener('token', (e) => {
    d.textContent += e.data;
  });
  es.addEventListener('end', () => { es.close(); sendBtn.disabled = false; });
};

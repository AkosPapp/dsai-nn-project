const formEl = document.getElementById('form');
const predEl = document.getElementById('pred');
let session = null;
let inputSize = 0;

async function loadModel() {
  session = await ort.InferenceSession.create('best_model.onnx');
  const r = await fetch('features.json');
  const data = await r.json();
  inputSize = data.input_size;
  buildTable(inputSize);
  await updatePrediction();
}

function buildTable(size) {
  const rows = Math.ceil(size / 4);
  const table = document.createElement('table');
  const thead = document.createElement('thead');
  const headRow = document.createElement('tr');
  ['index', 'time', 'x', 'y', 'z'].forEach((name) => {
    const th = document.createElement('th');
    th.textContent = name;
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);
  table.appendChild(thead);
  const tbody = document.createElement('tbody');
  for (let i = 0; i < rows; i += 1) {
    const row = document.createElement('tr');
    const indexCell = document.createElement('td');
    indexCell.setAttribute('data-label', 'index');
    indexCell.textContent = String(i);
    row.appendChild(indexCell);
    ['time', 'x', 'y', 'z'].forEach((label, colIndex) => {
      const td = document.createElement('td');
      td.setAttribute('data-label', label);
      const input = document.createElement('input');
      input.type = 'number';
      input.step = '0.01';
      input.value = '0';
      input.dataset.index = i * 4 + colIndex;
      if (Number(input.dataset.index) >= size) {
        input.disabled = true;
      }
      input.addEventListener('input', () => updatePrediction());
      td.appendChild(input);
      row.appendChild(td);
    });
    tbody.appendChild(row);
  }
  table.appendChild(tbody);
  formEl.innerHTML = '';
  formEl.appendChild(table);
}

async function updatePrediction() {
  if (!session) return;
  const inputs = Array.from(formEl.querySelectorAll('input'))
    .filter((el) => !el.disabled)
    .map((el) => Number(el.value));
  const inputTensor = new ort.Tensor('float32', Float32Array.from(inputs), [1, inputs.length]);
  const results = await session.run({ input: inputTensor });
  const outputName = Object.keys(results)[0];
  predEl.textContent = JSON.stringify(Array.from(results[outputName].data));
}

loadModel();

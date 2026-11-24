const url = '/api/todos';

async function request(path, method = 'GET', body) {
  let response;
  try {
    response = await fetch(path, body === undefined ? undefined : {
      method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
  } catch {
    throw new Error('暂时无法连接待办服务。');
  }
  const data = await response.json().catch(() => null);
  if (!response.ok || data === null) throw new Error(data?.error ?? '暂时无法读取或保存待办。');
  return data;
}

const panel = document.querySelector('[data-todo-summary], [data-todo-version]');
const feedback = panel.querySelector('[data-todo-feedback]');
const count = panel.querySelector('[data-todo-count]');

function showFeedback(message, error = false) {
  feedback.textContent = message;
  feedback.hidden = !message;
  feedback.classList.toggle('todo-error', error);
}

function attachRefresh(refresh) {
  window.addEventListener('pageshow', event => {
    if (event.persisted) refresh();
  });
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) refresh();
  });
  refresh();
}

function setupSummary() {
  const groups = [...panel.querySelectorAll('[data-todo-group]')];
  attachRefresh(async () => {
    try {
      const todos = await request(url);
      let total = 0;
      for (const group of groups) {
        const scope=(group.dataset.todoVersions??group.dataset.todoGroup).split(',');
        const codes=JSON.parse(group.dataset.todoCodes??'{}');
        const links=JSON.parse(group.dataset.todoLinks??'{}');
        const items = todos.filter(todo => scope.includes(todo.version));
        group.querySelector('[data-todo-items]').replaceChildren(...items.map(todo => {
          const item = document.createElement('li');
          const title=(codes[todo.version]?codes[todo.version]+' · ':'')+todo.text;
          if (links[todo.version]) {
            const link=document.createElement('a');link.href=links[todo.version];link.textContent=title;item.append(link);
          } else item.textContent=title;
          return item;
        }));
        group.hidden = items.length === 0;
        total += items.length;
      }
      count.textContent = total ? `${total} 项未完成` : '';
      showFeedback(total ? '' : '暂无未完成待办。');
    } catch (error) {
      showFeedback(error.message, true);
    }
  });
}

function setupVersion(version) {
  const form = panel.querySelector('[data-todo-form]');
  const fieldset = form.querySelector('fieldset');
  const text = form.elements.text;
  const list = panel.querySelector('[data-todo-items]');
  let todos = [];
  let busy = false;

  function setBusy(value) {
    busy = value;
    fieldset.disabled = value;
    for (const control of list.querySelectorAll('input, button')) control.disabled = value;
  }

  function render() {
    const done = todos.filter(todo => todo.completed).length;
    count.textContent = todos.length ? `${todos.length - done} 项待办 · ${done} 项已完成` : '';
    list.replaceChildren(...todos.sort((a, b) => a.completed - b.completed || a.id - b.id).map(todo => {
      const item = document.createElement('li');
      const label = document.createElement('label');
      const checkbox = document.createElement('input');
      const title = document.createElement('span');
      const remove = document.createElement('button');
      checkbox.type = 'checkbox';
      checkbox.dataset.todoId = todo.id;
      checkbox.checked = Boolean(todo.completed);
      title.textContent = todo.text;
      remove.type = 'button';
      remove.className = 'todo-delete';
      remove.dataset.deleteTodo = todo.id;
      remove.textContent = '删除';
      remove.setAttribute('aria-label', `删除待办 ${todo.text}`);
      label.append(checkbox, title);
      item.append(label, remove);
      return item;
    }));
  }

  async function save(method, fields) {
    if (busy) return;
    setBusy(true);
    try {
      const saved = await request(url, method, { ...fields, version });
      todos = todos.filter(todo => todo.id !== saved.id);
      if (method !== 'DELETE') todos.push(saved);
      if (method === 'POST') text.value = '';
      const feedbackByMethod = { POST: '已添加待办。', DELETE: '已删除待办。' };
      showFeedback(feedbackByMethod[method] ?? '');
    } catch (error) {
      showFeedback(error.message, true);
    } finally {
      render();
      setBusy(false);
      if (method === 'PATCH') list.querySelector(`[data-todo-id="${fields.id}"]`)?.focus({ preventScroll: true });
      else if (method === 'DELETE') (list.querySelector(`[data-delete-todo="${fields.id}"]`) ?? text).focus({ preventScroll: true });
      else text.focus();
    }
  }

  form.addEventListener('submit', event => {
    event.preventDefault();
    const value = text.value.trim();
    if (value) save('POST', { text: value });
    else showFeedback('请输入待办内容。', true);
  });

  list.addEventListener('change', event => {
    const input = event.target;
    save('PATCH', { id: Number(input.dataset.todoId), completed: input.checked });
  });

  list.addEventListener('click', event => {
    const remove = event.target.closest('[data-delete-todo]');
    if (remove && !busy && window.confirm('删除这条待办？')) {
      save('DELETE', { id: Number(remove.dataset.deleteTodo) });
    }
  });

  attachRefresh(async () => {
    if (busy) return;
    setBusy(true);
    try {
      todos = await request(`${url}?version=${encodeURIComponent(version)}`);
      render();
      showFeedback(todos.length ? '' : '这个版本暂无待办。');
    } catch (error) {
      showFeedback(error.message, true);
    } finally {
      setBusy(false);
    }
  });
}

if (panel.hasAttribute('data-todo-summary')) setupSummary();
else setupVersion(panel.dataset.todoVersion);

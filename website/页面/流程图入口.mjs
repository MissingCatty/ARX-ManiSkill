async function activate(widget) {
  const status = widget.closest('.workflow').querySelector('[data-workflow-state]');
  try {
    const css = document.createElement('link');
    css.rel = 'stylesheet';
    css.href = widget.dataset.stylesheet;
    const styled = new Promise((resolve, reject) => {
      css.onload = resolve;
      css.onerror = () => reject(new Error('流程图样式加载失败'));
    });
    document.head.append(css);
    const [module, response] = await Promise.all([import(widget.dataset.client), fetch(widget.dataset.source), styled]);
    if (!response.ok) throw new Error('流程图登记加载失败');
    await module.mountWorkflow(widget, await response.json());
    status.textContent = '点击卡片查看详情；拖动空白区域移动画布。';
  } catch (error) {
    widget.querySelector('[data-workflow-fallback]').hidden = false;
    widget.querySelector('[data-workflow-canvas]').hidden = true;
    status.textContent = '当前显示矢量流程图，点击节点仍可打开对应步骤页面。';
    console.error('流程图加载失败', error);
  }
}

for (const widget of document.querySelectorAll('[data-workflow-widget]')) {
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      if (!entries.some(entry => entry.isIntersecting)) return;
      observer.disconnect();
      activate(widget);
    }, {rootMargin: '200px'});
    observer.observe(widget);
  } else activate(widget);
}

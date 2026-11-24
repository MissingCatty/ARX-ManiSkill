import React, { Component, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Background, Controls, Handle, MarkerType, Panel, Position, ReactFlow } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import './交互流程图.css';

const colors = { fixed: '#b98a40', pending: '#969dab', normal: '#8498ad' };
const documentCache = new Map();

function StepNode({ data }) {
  const activate = event => {
    // Keep native link behaviour for new tabs and for browsers without this component.
    if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    event.stopPropagation();
    data.open(data.id);
  };
  return <>
    <Handle type="target" position={Position.Top} id="in-top" isConnectable={false} />
    <Handle type="target" position={Position.Left} id="in-left" isConnectable={false} />
    <Handle type="target" position={Position.Right} id="in-right" isConnectable={false} />
    <a href={data.href} onClick={activate} className={`workflow-diagram-node workflow-${data.kind} nodrag nopan`}
      data-workflow-step={data.id} aria-label={`${data.number} ${data.title}，查看详情`}>
      <span className="workflow-node-top"><span>{data.number}</span><span>{data.status}</span></span>
      <strong>{data.title}</strong><span className="workflow-node-summary">{data.summary}</span>
      <span className="workflow-node-open">查看详情 ↗</span>
    </a>
    <Handle type="source" position={Position.Bottom} id="out-bottom" isConnectable={false} />
    <Handle type="source" position={Position.Left} id="out-left" isConnectable={false} />
    <Handle type="source" position={Position.Right} id="out-right" isConnectable={false} />
  </>;
}
const nodeTypes = { step: StepNode };

function StepDetails({ step, close }) {
  const dialog = useRef(null);
  const [body, setBody] = useState(documentCache.get(step.href) ?? null);
  const [error, setError] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    const previousFocus = document.activeElement;
    dialog.current.showModal();
    if (!documentCache.has(step.href)) {
      fetch(step.href, { signal: controller.signal }).then(response => {
        if (!response.ok) throw new Error('步骤页面加载失败');
        return response.text();
      }).then(text => {
        // This is the same-origin, Markdown-generated article, not API-provided HTML.
        const article = new DOMParser().parseFromString(text, 'text/html').querySelector('.article > .markdown');
        if (!article) throw new Error('步骤正文缺失');
        documentCache.set(step.href, article.innerHTML);
        setBody(article.innerHTML);
      }).catch(error => { if (error.name !== 'AbortError') setError(true); });
    }
    return () => { controller.abort(); previousFocus?.focus({ preventScroll: true }); };
  }, [step.href]);
  return <dialog ref={dialog} className="workflow-dialog" aria-labelledby="workflow-detail-title"
    onCancel={event => { event.preventDefault(); close(); }} onClick={event => { if (event.target === dialog.current) close(); }}>
    <div className="workflow-dialog-inner">
      <header className="workflow-dialog-header">
        <div><p>步骤 {step.number} · {step.status}</p><h2 id="workflow-detail-title">{step.title}</h2></div>
        <button type="button" onClick={close} autoFocus aria-label="关闭步骤详情">关闭 ×</button>
      </header>
      <p className="workflow-dialog-summary">{step.summary}</p>
      {body ? <div className="markdown workflow-dialog-body" dangerouslySetInnerHTML={{ __html: body }} />
        : <p role="status">{error ? '暂时无法加载正文，请通过下方链接打开步骤页面。' : '正在加载步骤内容…'}</p>}
      <footer><a href={step.href}>打开独立步骤页 →</a><button type="button" onClick={close}>返回流程图</button></footer>
    </div>
  </dialog>;
}

function Workflow({ config, ready, widget }) {
  const [selected, setSelected] = useState(null);
  const instance = useRef(null);
  const open = useCallback(id => setSelected(config.steps.find(step => step.id === id)), [config]);
  const nodes = useMemo(() => config.steps.map(step => ({
    id: step.id, type: 'step', position: { x: config.graph.positions[step.id][0], y: config.graph.positions[step.id][1] },
    data: { ...step, open }, width: 280, height: 168, draggable: false, focusable: false,
  })), [config, open]);
  const edges = useMemo(() => config.graph.edges.map(edge => {
    const color = colors[edge.kind ?? 'normal'];
    return { ...edge, id: `${edge.source}-${edge.target}`, type: 'smoothstep',
      sourceHandle: edge.sourceHandle ?? 'out-bottom', targetHandle: edge.targetHandle ?? 'in-top',
      style: { stroke: color, strokeWidth: 2, ...(edge.kind ? { strokeDasharray: '6 5' } : {}) },
      markerEnd: { type: MarkerType.ArrowClosed, color },
      labelStyle: { fill: color, fontSize: 12 }, labelBgStyle: { fill: '#fbfcff', fillOpacity: 0.96 },
      labelBgPadding: [7, 4], labelBgBorderRadius: 4,
    };
  }), [config]);
  const start = useCallback(() => {
    const width = widget.clientWidth;
    const zoom = Math.max(0.4, Math.min(1, width < 520 ? (width - 24) / 280 : (width - 36) / 660));
    instance.current?.setViewport({ x: (width - 660 * zoom) / 2, y: 36, zoom }, { duration: 200 });
  }, [widget]);
  return <div className="workflow-app">
    <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} nodesDraggable={false}
      nodesConnectable={false} nodesFocusable={false} edgesFocusable={false} elementsSelectable={false}
      panOnDrag zoomOnScroll={false} zoomOnPinch zoomOnDoubleClick={false} preventScrolling={false}
      minZoom={0.18} maxZoom={1.7} deleteKeyCode={null} selectionKeyCode={null}
      onNodeClick={(_, node) => open(node.id)}
      onInit={flow => { instance.current = flow; start(); ready(); }} aria-label={config.title}>
      <Background gap={20} size={1} color="#dfe5f0" />
      <Controls showInteractive={false} fitViewOptions={{ padding: 0.1, duration: 250 }} aria-label="流程图缩放" />
      <Panel position="top-right" className="workflow-tools">
        <button type="button" onClick={() => instance.current?.fitView({ padding: 0.1, duration: 250 })}>全图</button>
        <button type="button" onClick={start}>回到起点</button>
        <button type="button" onClick={() => {
          if (document.fullscreenElement) document.exitFullscreen?.();
          else widget.requestFullscreen?.().catch(() => {});
        }}>全屏</button>
      </Panel>
    </ReactFlow>
    {selected && <StepDetails key={selected.id} step={selected} close={() => setSelected(null)} />}
  </div>;
}

class WorkflowBoundary extends Component {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  componentDidCatch(error) { this.props.fail(error); }
  render() { return this.state.failed ? null : this.props.children; }
}

export function mountWorkflow(widget, config) {
  const host = widget.querySelector('[data-workflow-canvas]');
  const fallback = widget.querySelector('[data-workflow-fallback]');
  host.hidden = false;
  // Keep the static graph visible until the interactive canvas has measured its nodes.
  const root = createRoot(host);
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { root.unmount(); reject(new Error('流程图初始化超时')); }, 10000);
    const ready = () => { clearTimeout(timer); fallback.hidden = true; resolve(); };
    const fail = error => { clearTimeout(timer); fallback.hidden = false; host.hidden = true; reject(error); };
    root.render(<WorkflowBoundary fail={fail}><Workflow config={config} ready={ready} widget={widget} /></WorkflowBoundary>);
  });
}

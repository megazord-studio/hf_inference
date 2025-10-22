(function () {
  const canvas = document.getElementById('net-bg');
  if (!canvas) return;

  const ctx = canvas.getContext('2d', {alpha: true});
  const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
  let width = 0, height = 0;
  let nodes = [];
  let running = true;

  // Respect reduced motion
  const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  const isReduced = () => prefersReduced.matches;

  function getVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function theme() {
    const p = getVar('--p') || '200 90% 58%';
    const a = getVar('--a') || '265 85% 68%';
    return {
      link: (alpha) => `hsl(${p} / ${alpha})`,
      node: (alpha) => `hsl(${a} / ${alpha})`,
    };
  }

  function resize() {
    width = Math.floor(window.innerWidth);
    height = Math.floor(window.innerHeight);
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    initNodes();
  }

  function initNodes() {
    // Density-based count (kept modest for perf)
    const area = width * height;
    const target = Math.max(30, Math.min(110, Math.round(area / 26000)));
    const speed = 0.10 + Math.min(0.20, area / 1.5e6); // px/frame (faster)

    const newNodes = [];
    for (let i = 0; i < target; i++) {
      newNodes.push({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * speed,
        vy: (Math.random() - 0.5) * speed,
        r: 1.3 + Math.random(),
      });
    }
    nodes = newNodes;
  }

  function step() {
    const th = theme();
    const linkBaseAlpha = 0.22; // brighter lines
    const nodeAlpha = 0.58; // brighter nodes
    const linkDist = Math.max(110, Math.min(180, Math.hypot(width, height) * 0.10));
    const linkDist2 = linkDist * linkDist;

    ctx.clearRect(0, 0, width, height);

    // Update positions (wrap edges)
    for (let n of nodes) {
      n.x += n.vx;
      n.y += n.vy;
      if (n.x < -5) n.x = width + 5; else if (n.x > width + 5) n.x = -5;
      if (n.y < -5) n.y = height + 5; else if (n.y > height + 5) n.y = -5;
    }

    // Track nearest neighbor to ensure connectivity
    const nearestIdx = new Array(nodes.length).fill(-1);
    const nearestD2 = new Array(nodes.length).fill(Infinity);

    // Draw links within distance and record nearest neighbors
    ctx.lineWidth = 1.25;
    for (let i = 0; i < nodes.length; i++) {
      const ni = nodes[i];
      for (let j = i + 1; j < nodes.length; j++) {
        const nj = nodes[j];
        const dx = ni.x - nj.x;
        const dy = ni.y - nj.y;
        const d2 = dx * dx + dy * dy;

        if (d2 < nearestD2[i]) {
          nearestD2[i] = d2;
          nearestIdx[i] = j;
        }
        if (d2 < nearestD2[j]) {
          nearestD2[j] = d2;
          nearestIdx[j] = i;
        }

        if (d2 < linkDist2) {
          const a = linkBaseAlpha * (1 - d2 / linkDist2);
          ctx.strokeStyle = th.link(a.toFixed(3));
          ctx.beginPath();
          ctx.moveTo(ni.x, ni.y);
          ctx.lineTo(nj.x, nj.y);
          ctx.stroke();
        }
      }
    }

    // Ensure every node connects to its nearest neighbor for a cohesive net
    for (let i = 0; i < nodes.length; i++) {
      const j = nearestIdx[i];
      if (j > i && j !== -1) {
        // Slightly stronger than the weakest distance lines
        const a = Math.max(0.18, linkBaseAlpha * 0.9);
        ctx.strokeStyle = th.link(a.toFixed(3));
        ctx.beginPath();
        ctx.moveTo(nodes[i].x, nodes[i].y);
        ctx.lineTo(nodes[j].x, nodes[j].y);
        ctx.stroke();
      }
    }

    // Draw nodes
    for (let n of nodes) {
      ctx.fillStyle = th.node(nodeAlpha);
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function loop() {
    if (!running) return;
    step();
    requestAnimationFrame(loop);
  }

  function renderStatic() {
    // One-time render without motion, denser but gentle
    const th = theme();
    ctx.clearRect(0, 0, width, height);

    // Slightly increase link density for static
    const linkDist = Math.max(120, Math.min(190, Math.hypot(width, height) * 0.11));
    const linkDist2 = linkDist * linkDist;

    // Track nearest neighbor to ensure connectivity
    const nearestIdx = new Array(nodes.length).fill(-1);
    const nearestD2 = new Array(nodes.length).fill(Infinity);

    // Links
    ctx.lineWidth = 1.2;
    for (let i = 0; i < nodes.length; i++) {
      const ni = nodes[i];
      for (let j = i + 1; j < nodes.length; j++) {
        const nj = nodes[j];
        const dx = ni.x - nj.x;
        const dy = ni.y - nj.y;
        const d2 = dx * dx + dy * dy;
        if (d2 < nearestD2[i]) {
          nearestD2[i] = d2;
          nearestIdx[i] = j;
        }
        if (d2 < nearestD2[j]) {
          nearestD2[j] = d2;
          nearestIdx[j] = i;
        }
        if (d2 < linkDist2) {
          const a = 0.18 * (1 - d2 / linkDist2);
          ctx.strokeStyle = th.link(a.toFixed(3));
          ctx.beginPath();
          ctx.moveTo(ni.x, ni.y);
          ctx.lineTo(nj.x, nj.y);
          ctx.stroke();
        }
      }
    }

    // Ensure nearest neighbor connection
    for (let i = 0; i < nodes.length; i++) {
      const j = nearestIdx[i];
      if (j > i && j !== -1) {
        ctx.strokeStyle = th.link(0.22);
        ctx.beginPath();
        ctx.moveTo(nodes[i].x, nodes[i].y);
        ctx.lineTo(nodes[j].x, nodes[j].y);
        ctx.stroke();
      }
    }

    // Nodes
    for (let n of nodes) {
      ctx.fillStyle = th.node(0.55);
      ctx.beginPath();
      ctx.arc(n.x, n.y, Math.max(1.3, n.r), 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Start / Stop on visibility
  document.addEventListener('visibilitychange', () => {
    running = !document.hidden && !isReduced();
    if (running) requestAnimationFrame(loop);
  });

  // Resize
  window.addEventListener('resize', resize, {passive: true});
  prefersReduced.addEventListener('change', () => {
    if (isReduced()) {
      running = false;
      renderStatic();
    } else {
      running = true;
      requestAnimationFrame(loop);
    }
  });

  // Init
  resize();
  if (isReduced()) {
    renderStatic();
  } else {
    requestAnimationFrame(loop);
  }
})();

const MANIFEST_URL = 'static/gallery/v2/manifest.json';

function createImageTile(column, imagePath, score) {
  const tile = document.createElement('div');
  tile.className = 'image-tile';
  if (column.key === 'ours' || column.key === 'full') {
    tile.classList.add('image-tile--ours');
  }

  const label = document.createElement('div');
  label.className = 'image-tile-label';
  label.textContent = column.label;
  tile.appendChild(label);

  const link = document.createElement('a');
  link.href = imagePath;
  link.target = '_blank';
  link.rel = 'noopener noreferrer';

  const image = document.createElement('img');
  image.src = imagePath;
  image.alt = `${column.label} restoration panel`;
  image.loading = 'lazy';
  image.decoding = 'async';
  link.appendChild(image);
  tile.appendChild(link);

  if (score && typeof score.value === 'number') {
    const scoreNode = document.createElement('div');
    scoreNode.className = 'image-tile-score';
    scoreNode.textContent = `${score.label || 'Score'} ${score.value.toFixed(3)}`;
    tile.appendChild(scoreNode);
  }

  return tile;
}

function createCaseCard(dataset, item) {
  const card = document.createElement('article');
  card.className = 'case-card';

  const title = document.createElement('h4');
  title.className = 'case-card-title';
  title.textContent = `case ${item.sample_id}`;
  card.appendChild(title);

  const wrap = document.createElement('div');
  wrap.className = 'case-strip-wrap';

  const strip = document.createElement('div');
  strip.className = 'case-strip';
  strip.style.setProperty('--columns', String(dataset.columns.length));

  dataset.columns.forEach((column) => {
    const imagePath = item.images[column.key];
    const score = item.scores ? item.scores[column.key] : null;
    strip.appendChild(createImageTile(column, imagePath, score));
  });

  wrap.appendChild(strip);
  card.appendChild(wrap);
  return card;
}

function renderDatasetContent(details, dataset) {
  if (details.dataset.rendered === 'true') return;

  const content = document.createElement('div');
  content.className = 'dataset-content';
  dataset.cases.forEach((item) => content.appendChild(createCaseCard(dataset, item)));
  details.appendChild(content);
  details.dataset.rendered = 'true';
}

function createDatasetBlock(dataset) {
  const details = document.createElement('details');
  details.className = 'dataset-block';

  const summary = document.createElement('summary');
  const textWrap = document.createElement('span');
  textWrap.className = 'dataset-summary-text';

  const title = document.createElement('span');
  title.className = 'dataset-summary-title';
  title.textContent = dataset.title;
  textWrap.appendChild(title);

  const description = document.createElement('span');
  description.className = 'dataset-summary-description';
  description.textContent = dataset.description;
  textWrap.appendChild(description);

  const meta = document.createElement('span');
  meta.className = 'dataset-summary-meta';
  meta.textContent = `${dataset.cases.length} ${dataset.cases.length === 1 ? 'case' : 'cases'}`;

  summary.appendChild(textWrap);
  summary.appendChild(meta);
  details.appendChild(summary);

  details.addEventListener('toggle', () => {
    if (details.open) renderDatasetContent(details, dataset);
  });

  if (dataset.open_by_default) {
    details.open = true;
    renderDatasetContent(details, dataset);
  }

  return details;
}

function createGalleryGroup(group) {
  const section = document.createElement('section');
  section.className = 'gallery-group';

  const header = document.createElement('div');
  header.className = 'gallery-group-header';

  const title = document.createElement('h3');
  title.textContent = group.title;
  header.appendChild(title);

  const description = document.createElement('p');
  description.textContent = group.description;
  header.appendChild(description);
  section.appendChild(header);

  group.datasets.forEach((dataset) => section.appendChild(createDatasetBlock(dataset)));
  return section;
}

function renderGalleryRoots(manifest) {
  const groupMap = new Map(manifest.groups.map((group) => [group.id, group]));
  document.querySelectorAll('[data-gallery-groups]').forEach((root) => {
    const groupIds = root.dataset.galleryGroups.split(',').map((value) => value.trim()).filter(Boolean);
    groupIds.forEach((groupId) => {
      const group = groupMap.get(groupId);
      if (group) root.appendChild(createGalleryGroup(group));
    });
  });
}

async function loadGalleries() {
  try {
    const response = await fetch(MANIFEST_URL);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const manifest = await response.json();
    renderGalleryRoots(manifest);
  } catch (error) {
    document.querySelectorAll('[data-gallery-groups]').forEach((root) => {
      const message = document.createElement('div');
      message.className = 'gallery-error';
      message.textContent = 'The visual gallery could not be loaded. Please refresh the page.';
      root.appendChild(message);
    });
    console.error('Failed to load IConFace gallery manifest:', error);
  }
}

function setupLazyFigures() {
  document.querySelectorAll('[data-lazy-figure]').forEach((details) => {
    details.addEventListener('toggle', () => {
      if (!details.open) return;
      details.querySelectorAll('img[data-src]').forEach((image) => {
        image.src = image.dataset.src;
        image.removeAttribute('data-src');
      });
    });
  });
}

function setupBibTeXCopy() {
  const button = document.querySelector('.copy-bibtex-btn');
  const code = document.getElementById('bibtex-code');
  if (!button || !code) return;

  button.addEventListener('click', async () => {
    const text = code.textContent;
    try {
      await navigator.clipboard.writeText(text);
    } catch (_) {
      const area = document.createElement('textarea');
      area.value = text;
      document.body.appendChild(area);
      area.select();
      document.execCommand('copy');
      area.remove();
    }

    const label = button.querySelector('.copy-text');
    button.classList.add('copied');
    label.textContent = 'Copied';
    window.setTimeout(() => {
      button.classList.remove('copied');
      label.textContent = 'Copy';
    }, 1800);
  });
}

function setupScrollToTop() {
  const button = document.querySelector('.scroll-to-top');
  if (!button) return;

  const updateVisibility = () => button.classList.toggle('visible', window.scrollY > 500);
  window.addEventListener('scroll', updateVisibility, { passive: true });
  button.addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));
  updateVisibility();
}

document.addEventListener('DOMContentLoaded', () => {
  setupBibTeXCopy();
  setupScrollToTop();
  setupLazyFigures();
  loadGalleries();
});

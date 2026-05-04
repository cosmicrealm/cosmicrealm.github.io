const LABEL_MAP = {
  ref1: 'Ref 1',
  ref2: 'Ref 2',
  ref3: 'Ref 3',
  lq: 'LQ',
  deg: 'LQ',
  gt: 'GT',
  ours: 'Ours',
  dmdnet: 'DMDNet',
  refldm: 'ReF-LDM',
  restorerid: 'RestorerID',
  instantrestore: 'InstantRestore',
  faceme: 'FaceMe',
  codeformer: 'CodeFormer',
  gfpgan: 'GFP-GAN',
  restoreformerpp: 'RestoreFormer++',
  vqfr: 'VQFR',
  daefr: 'DAEFR',
  concat: 'Concat',
  struct: 'Struct',
  id: 'ID',
  '1r': '1-Route',
  full: 'Full'
};

function copyBibTeX() {
  const bibtexElement = document.getElementById('bibtex-code');
  const button = document.querySelector('.copy-bibtex-btn');
  const copyText = button ? button.querySelector('.copy-text') : null;

  if (!bibtexElement || !button || !copyText) return;

  navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
    button.classList.add('copied');
    copyText.textContent = 'Copied';
    setTimeout(function() {
      button.classList.remove('copied');
      copyText.textContent = 'Copy';
    }, 2000);
  }).catch(function() {
    const textArea = document.createElement('textarea');
    textArea.value = bibtexElement.textContent;
    document.body.appendChild(textArea);
    textArea.select();
    document.execCommand('copy');
    document.body.removeChild(textArea);

    button.classList.add('copied');
    copyText.textContent = 'Copied';
    setTimeout(function() {
      button.classList.remove('copied');
      copyText.textContent = 'Copy';
    }, 2000);
  });
}

function scrollToTop() {
  window.scrollTo({
    top: 0,
    behavior: 'smooth'
  });
}

function formatColumnLabel(key) {
  return LABEL_MAP[key] || key;
}

function computeDatasetWidthPx(columnCount, galleryType) {
  const gapPx = 12.8;
  const paddingPx = 32;
  let tileWidthPx = 150;

  if (galleryType === 'blind') {
    tileWidthPx = columnCount >= 8 ? 148 : 146;
  } else if (galleryType === 'ablation') {
    tileWidthPx = 150;
  } else {
    tileWidthPx = 150;
  }

  return Math.round(columnCount * tileWidthPx + Math.max(0, columnCount - 1) * gapPx + paddingPx);
}

function getDisplayColumns(dataset) {
  if (dataset.columns.includes('ref2') || dataset.columns.includes('ref3')) {
    return dataset.columns.filter((key) => key !== 'ref2' && key !== 'ref3');
  }
  return dataset.columns;
}

function createImageTile(label, imagePath, score) {
  const tile = document.createElement('div');
  tile.className = imagePath ? 'image-tile' : 'image-tile image-tile--missing';

  const tileLabel = document.createElement('div');
  tileLabel.className = 'image-tile-label';
  tileLabel.textContent = label;
  tile.appendChild(tileLabel);

  if (imagePath) {
    const link = document.createElement('a');
    link.href = imagePath;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';

    const img = document.createElement('img');
    img.src = imagePath;
    img.alt = label;
    img.loading = 'lazy';
    img.decoding = 'async';
    link.appendChild(img);
    tile.appendChild(link);
  } else {
    const placeholder = document.createElement('div');
    placeholder.className = 'image-tile-placeholder';
    placeholder.textContent = 'N/A';
    tile.appendChild(placeholder);
  }

  if (typeof score === 'number') {
    const scoreNode = document.createElement('div');
    scoreNode.className = 'image-tile-score';
    scoreNode.textContent = `R1 ${score.toFixed(3)}`;
    tile.appendChild(scoreNode);
  } else if (score && typeof score === 'object') {
    const scoreNode = document.createElement('div');
    scoreNode.className = 'image-tile-score image-tile-score--stacked';

    if (typeof score.gt === 'number') {
      const gtLine = document.createElement('span');
      gtLine.textContent = `GT ${score.gt.toFixed(3)}`;
      scoreNode.appendChild(gtLine);
    }

    if (typeof score.r1 === 'number') {
      const r1Line = document.createElement('span');
      r1Line.textContent = `R1 ${score.r1.toFixed(3)}`;
      scoreNode.appendChild(r1Line);
    }

    tile.appendChild(scoreNode);
  }

  return tile;
}

function createCaseCard(dataset, item, index) {
  const card = document.createElement('article');
  card.className = 'case-card';

  const header = document.createElement('div');
  header.className = 'case-card-header';

  const title = document.createElement('h4');
  title.className = 'case-card-title';
  title.textContent = `#${index + 1} · ${item.sample_id}`;
  header.appendChild(title);

  if (typeof item.record_index === 'number') {
    const meta = document.createElement('div');
    meta.className = 'case-card-meta';
    meta.textContent = `record ${item.record_index}`;
    header.appendChild(meta);
  }

  card.appendChild(header);

  const strip = document.createElement('div');
  strip.className = 'case-strip';
  const displayColumns = getDisplayColumns(dataset);

  displayColumns.forEach((columnKey) => {
    const imagePath = item.images ? item.images[columnKey] : null;
    const score = item.scores ? item.scores[columnKey] : null;
    strip.appendChild(createImageTile(formatColumnLabel(columnKey), imagePath, score));
  });

  card.appendChild(strip);
  return card;
}

function createDatasetBlock(dataset, openByDefault, galleryType) {
  const details = document.createElement('details');
  details.className = 'dataset-block';
  const displayColumns = getDisplayColumns(dataset);
  details.style.setProperty('--dataset-columns', String(displayColumns.length));
  details.style.width = `min(100%, ${computeDatasetWidthPx(displayColumns.length, galleryType)}px)`;
  if (openByDefault) {
    details.open = true;
  }

  const summary = document.createElement('summary');
  summary.className = 'dataset-summary';

  const titleWrap = document.createElement('div');
  titleWrap.className = 'dataset-summary-text';

  const title = document.createElement('h3');
  title.className = 'dataset-title';
  title.textContent = dataset.name;
  titleWrap.appendChild(title);

  const meta = document.createElement('p');
  meta.className = 'dataset-meta';
  meta.textContent = `${dataset.cases.length} cases · ${displayColumns.map(formatColumnLabel).join(' / ')}`;
  titleWrap.appendChild(meta);

  summary.appendChild(titleWrap);
  details.appendChild(summary);

  if (dataset.description) {
    const description = document.createElement('p');
    description.className = 'dataset-description';
    description.textContent = dataset.description;
    details.appendChild(description);
  }

  const caseList = document.createElement('div');
  caseList.className = 'case-list';
  dataset.cases.forEach((item, index) => {
    caseList.appendChild(createCaseCard(dataset, item, index));
  });

  details.appendChild(caseList);
  return details;
}

function renderGallerySection(rootId, datasets, galleryType) {
  const root = document.getElementById(rootId);
  if (!root) return;

  root.innerHTML = '';
  datasets.forEach((dataset, index) => {
    root.appendChild(createDatasetBlock(dataset, index === 0, galleryType));
  });
}

function renderError(rootId, message) {
  const root = document.getElementById(rootId);
  if (!root) return;
  const error = document.createElement('div');
  error.className = 'gallery-error';
  error.textContent = message;
  root.innerHTML = '';
  root.appendChild(error);
}

async function loadGalleries() {
  try {
    const [paperManifestResp] = await Promise.all([
      fetch('static/gallery/paper/manifest.json')
    ]);

    if (!paperManifestResp.ok) {
      throw new Error(`Failed to load paper manifest: ${paperManifestResp.status}`);
    }

    const paperManifest = await paperManifestResp.json();

    renderGallerySection('main-paper-figures-root', paperManifest.main_datasets || [], 'paper');
    renderGallerySection('supplementary-figures-root', paperManifest.supplementary_datasets || [], 'paper');
  } catch (error) {
    console.error(error);
    renderError('main-paper-figures-root', 'Failed to load main-paper figure assets.');
    renderError('supplementary-figures-root', 'Failed to load supplementary figure assets.');
  }
}

window.addEventListener('scroll', function() {
  const scrollButton = document.querySelector('.scroll-to-top');
  if (!scrollButton) return;

  if (window.pageYOffset > 300) {
    scrollButton.classList.add('visible');
  } else {
    scrollButton.classList.remove('visible');
  }
});

window.addEventListener('DOMContentLoaded', function() {
  loadGalleries();
});

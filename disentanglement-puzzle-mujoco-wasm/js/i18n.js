const MESSAGES = {
  ja: {
    header: {
      title: '位相ベース 3D 知恵の輪',
      subtitle: 'Loop / Gap の内在表現から alpha 系のワイヤー知恵の輪を生成',
      language: 'Language',
    },
    status: {
      loading: 'Loading MuJoCo…',
      compiling: 'モデルをコンパイル中…',
      ready: 'MuJoCo Ready',
      loadError: 'Load Error',
      compileError: 'Compile Error',
    },
    mode: {
      notStarted: '未開始',
      initializing: '初期化中',
      generating: '生成中',
      playing: 'プレイ中',
      paused: '一時停止',
      solved: 'クリア',
      error: 'エラー',
    },
    section: {
      settings: '生成設定',
      playState: 'プレイ状態',
      objective: '目的',
      diagnostics: '診断 / デバッグ',
      controls: '操作方法',
      log: 'イベントログ',
    },
    field: {
      complexity: '複雑性',
      seed: 'Seed',
    },
    button: {
      random: 'ランダム',
      generate: '新しい知恵の輪を生成',
      reset: '初期位置へ戻す',
      pause: '一時停止',
      resume: '再開',
      copyUrl: 'URL をコピー',
      recomputeDiagnostics: '診断を再計算',
      downloadDiagnostics: '診断 JSON',
      downloadSpec: 'spec JSON',
      downloadMjcf: 'MJCF XML',
      copyLog: 'ログをコピー',
    },
    toggle: {
      debugMarkers: '診断マーカー表示',
      followCamera: 'カメラ追従',
      shiftTurbo: 'Shift で加速',
    },
    stat: {
      family: 'family',
      gapRatio: 'gap / 線径比',
      difficulty: '推定難度',
      wireLength: '総ワイヤー長',
      separation: '分離度',
      contactCount: '接触数',
      elapsedTime: '経過時間',
      currentSeed: '現在の Seed',
      mode: 'モード',
    },
    objective: {
      text: 'シアン色の可動片を銀色の固定片から分離してください。パズルは gap align → twist → slide の手順を持つ alpha 系の機構として生成されます。',
    },
    diag: {
      staticTitle: '静的診断',
      runtimeTitle: '実行時診断',
    },
    controls: {
      movement: '移動',
      rotation: '回転',
      view: 'ビュー',
      other: 'その他',
      moveWasd: 'カメラ基準で前後左右',
      moveRf: '上 / 下',
      moveShift: '推力を増やす',
      rotateIk: 'ピッチ',
      rotateJl: 'ヨー',
      rotateUo: 'ロール',
      viewMouse: 'カメラ回転',
      viewWheel: 'ズーム',
      viewC: 'カメラ再センタリング',
      otherSpace: '一時停止 / 再開',
      otherBackspace: 'リセット',
      otherM: '診断マーカー切替',
    },
    log: {
      waiting: '起動待ち…',
      mujocoInitialized: 'MuJoCo WASM を初期化しました。source={{source}}',
      mujocoApiSummary: 'MuJoCo API: staticLoad={{staticLoad}}, legacyModel={{legacyModel}}, mjData={{mjData}}, mj_step={{mjStep}}',
      mujocoInitFailed: 'MuJoCo 初期化失敗: {{message}}',
      generatePuzzle: 'パズル生成: family={{family}}, seed={{seed}}, complexity={{complexity}}',
      generateFailed: '生成失敗: {{message}}',
      reset: 'パズルを初期状態に戻しました。',
      diagnosticsRecomputed: '静的診断と実行診断を再計算しました。',
      downloadedDiagnostics: '診断 JSON をダウンロードしました。',
      downloadedSpec: 'spec JSON をダウンロードしました。',
      downloadedMjcf: 'MJCF XML をダウンロードしました。',
      selfCheckSummary: '自己診断: {{total}} ケース中 {{badCount}} 件の要確認。',
      solved: 'クリア: {{time}} seed={{seed}} clearance={{clearance}} threading={{threading}}',
      windowError: 'window.onerror: {{message}}',
      unhandledRejection: 'unhandledrejection: {{reason}}',
    },
    overlay: {
      generated: '新しい知恵の輪を生成しました。シアンの可動片を分離してください。',
      generateFailed: 'パズル生成に失敗しました。\n\n{{message}}',
      resetDone: '初期位置へ戻しました。',
      urlCopied: '現在の URL をコピーしました。',
      diagnosticsRecomputed: '診断を再計算しました。',
      logCopied: 'ログをコピーしました。',
      solved: 'クリア！ 経過時間: {{time}} family: {{family}} seed: {{seed}}',
      initFailed: 'MuJoCo の初期化に失敗しました。\n\n{{message}}',
    },
    prompt: {
      copyUrl: 'URL をコピーしてください',
      copyLog: 'ログをコピーしてください',
    },
  },
  en: {
    header: {
      title: 'Topology-Based 3D Wire Puzzle',
      subtitle: 'Generate alpha-style disentanglement puzzles from intrinsic Loop / Gap representations',
      language: 'Language',
    },
    status: {
      loading: 'Loading MuJoCo…',
      compiling: 'Compiling model…',
      ready: 'MuJoCo Ready',
      loadError: 'Load Error',
      compileError: 'Compile Error',
    },
    mode: {
      notStarted: 'Not started',
      initializing: 'Initializing',
      generating: 'Generating',
      playing: 'Playing',
      paused: 'Paused',
      solved: 'Solved',
      error: 'Error',
    },
    section: {
      settings: 'Generation Settings',
      playState: 'Play State',
      objective: 'Objective',
      diagnostics: 'Diagnostics / Debug',
      controls: 'Controls',
      log: 'Event Log',
    },
    field: {
      complexity: 'Complexity',
      seed: 'Seed',
    },
    button: {
      random: 'Random',
      generate: 'Generate New Puzzle',
      reset: 'Reset to Start',
      pause: 'Pause',
      resume: 'Resume',
      copyUrl: 'Copy URL',
      recomputeDiagnostics: 'Recompute Diagnostics',
      downloadDiagnostics: 'Diagnostics JSON',
      downloadSpec: 'spec JSON',
      downloadMjcf: 'MJCF XML',
      copyLog: 'Copy Log',
    },
    toggle: {
      debugMarkers: 'Show diagnostic markers',
      followCamera: 'Follow camera target',
      shiftTurbo: 'Shift boosts thrust',
    },
    stat: {
      family: 'Family',
      gapRatio: 'Gap / wire ratio',
      difficulty: 'Estimated difficulty',
      wireLength: 'Total wire length',
      separation: 'Separation',
      contactCount: 'Contact count',
      elapsedTime: 'Elapsed time',
      currentSeed: 'Current seed',
      mode: 'Mode',
    },
    objective: {
      text: 'Separate the cyan moving piece from the silver fixed piece. Each puzzle is generated as an alpha-style mechanism with a gap align → twist → slide solution path.',
    },
    diag: {
      staticTitle: 'Static Diagnostics',
      runtimeTitle: 'Runtime Diagnostics',
    },
    controls: {
      movement: 'Movement',
      rotation: 'Rotation',
      view: 'View',
      other: 'Other',
      moveWasd: 'forward / left / back / right in camera space',
      moveRf: 'up / down',
      moveShift: 'increase thrust',
      rotateIk: 'pitch',
      rotateJl: 'yaw',
      rotateUo: 'roll',
      viewMouse: 'orbit camera',
      viewWheel: 'zoom',
      viewC: 'recenter camera',
      otherSpace: 'pause / resume',
      otherBackspace: 'reset',
      otherM: 'toggle diagnostic markers',
    },
    log: {
      waiting: 'Waiting to boot…',
      mujocoInitialized: 'Initialized MuJoCo WASM. source={{source}}',
      mujocoApiSummary: 'MuJoCo API: staticLoad={{staticLoad}}, legacyModel={{legacyModel}}, mjData={{mjData}}, mj_step={{mjStep}}',
      mujocoInitFailed: 'MuJoCo initialization failed: {{message}}',
      generatePuzzle: 'Generated puzzle: family={{family}}, seed={{seed}}, complexity={{complexity}}',
      generateFailed: 'Generation failed: {{message}}',
      reset: 'Reset the puzzle to its initial state.',
      diagnosticsRecomputed: 'Recomputed static and runtime diagnostics.',
      downloadedDiagnostics: 'Downloaded diagnostics JSON.',
      downloadedSpec: 'Downloaded spec JSON.',
      downloadedMjcf: 'Downloaded MJCF XML.',
      selfCheckSummary: 'Self-check: {{badCount}} flagged cases out of {{total}}.',
      solved: 'Solved: {{time}} seed={{seed}} clearance={{clearance}} threading={{threading}}',
      windowError: 'window.onerror: {{message}}',
      unhandledRejection: 'unhandledrejection: {{reason}}',
    },
    overlay: {
      generated: 'Generated a new wire puzzle. Separate the cyan moving piece.',
      generateFailed: 'Failed to generate the puzzle.\n\n{{message}}',
      resetDone: 'Reset to the initial pose.',
      urlCopied: 'Copied the current URL.',
      diagnosticsRecomputed: 'Recomputed diagnostics.',
      logCopied: 'Copied the event log.',
      solved: 'Solved! Elapsed time: {{time}} family: {{family}} seed: {{seed}}',
      initFailed: 'Failed to initialize MuJoCo.\n\n{{message}}',
    },
    prompt: {
      copyUrl: 'Copy this URL',
      copyLog: 'Copy this log text',
    },
  },
};

const SUPPORTED_LOCALES = ['ja', 'en'];

function getByPath(object, path) {
  return path.split('.').reduce((value, segment) => value?.[segment], object);
}

function interpolate(template, params) {
  return String(template).replace(/\{\{(\w+)\}\}/g, (_, key) => {
    const value = params[key];
    return value === undefined || value === null ? '' : String(value);
  });
}

export function normalizeLocale(locale) {
  if (!locale) return 'ja';
  const lowered = String(locale).toLowerCase();
  if (lowered.startsWith('en')) return 'en';
  return 'ja';
}

export function detectInitialLocale() {
  try {
    const params = new URLSearchParams(window.location.search);
    const fromQuery = normalizeLocale(params.get('lang'));
    if (SUPPORTED_LOCALES.includes(fromQuery)) return fromQuery;
  } catch {
    // ignore
  }

  try {
    const stored = normalizeLocale(window.localStorage.getItem('chie-no-wa.locale'));
    if (SUPPORTED_LOCALES.includes(stored)) return stored;
  } catch {
    // ignore
  }

  try {
    return normalizeLocale(window.navigator.language);
  } catch {
    return 'ja';
  }
}

export function createI18n(initialLocale = 'ja') {
  let locale = normalizeLocale(initialLocale);
  const listeners = new Set();

  function t(path, params = {}) {
    const entry =
      getByPath(MESSAGES[locale], path) ??
      getByPath(MESSAGES.ja, path) ??
      path;

    if (typeof entry === 'function') {
      return entry(params);
    }

    return interpolate(entry, params);
  }

  function apply(root = document) {
    root.querySelectorAll('[data-i18n]').forEach((node) => {
      node.textContent = t(node.dataset.i18n);
    });

    root.querySelectorAll('[data-i18n-placeholder]').forEach((node) => {
      node.setAttribute('placeholder', t(node.dataset.i18nPlaceholder));
    });

    root.querySelectorAll('[data-i18n-title]').forEach((node) => {
      node.setAttribute('title', t(node.dataset.i18nTitle));
    });
  }

  function setLocale(nextLocale) {
    const normalized = normalizeLocale(nextLocale);
    if (locale === normalized) return;

    locale = normalized;

    try {
      window.localStorage.setItem('chie-no-wa.locale', locale);
    } catch {
      // ignore storage errors
    }

    listeners.forEach((listener) => listener(locale));
  }

  function onChange(listener) {
    listeners.add(listener);
    return () => listeners.delete(listener);
  }

  return {
    get locale() {
      return locale;
    },
    t,
    apply,
    setLocale,
    onChange,
  };
}

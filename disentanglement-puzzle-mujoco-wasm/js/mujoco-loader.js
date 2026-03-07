const MUJOCO_CANDIDATE_URLS = [
  'https://cdn.jsdelivr.net/npm/mujoco-js@0.0.7/dist/mujoco_wasm.js',
  'https://unpkg.com/mujoco-js@0.0.7/dist/mujoco_wasm.js',
];

async function importWithTimeout(url, timeoutMs = 20000) {
  let timer = null;

  try {
    return await Promise.race([
      import(url),
      new Promise((_, reject) => {
        timer = setTimeout(() => {
          reject(new Error(`Timed out while importing ${url}`));
        }, timeoutMs);
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

export async function loadMujocoBindings(log = () => {}) {
  const failures = [];

  for (const url of MUJOCO_CANDIDATE_URLS) {
    try {
      log(`Loading MuJoCo module: ${url}`);
      const module = await importWithTimeout(url, 20000);
      const loader = module?.default;

      if (typeof loader !== 'function') {
        throw new Error('The module default export is not a function.');
      }

      log(`Initializing MuJoCo loader: ${url}`);

      const mujoco = await Promise.race([
        loader(),
        new Promise((_, reject) =>
          setTimeout(() => reject(new Error(`Timed out while initializing MuJoCo from ${url}`)), 30000),
        ),
      ]);

      return {
        mujoco,
        sourceUrl: url,
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      failures.push(`${url} :: ${message}`);
      log(`MuJoCo candidate failed: ${url} :: ${message}`, 'warn');
    }
  }

  throw new Error(`Failed to load MuJoCo module. ${failures.join(' | ')}`);
}

export function summarizeMujocoApi(mujoco) {
  return {
    topLevelKeys: Object.keys(mujoco).filter((key) => !key.startsWith('_')).sort(),
    hasFS: !!mujoco?.FS,
    hasMEMFS: !!mujoco?.MEMFS,
    hasMjModelClass: !!mujoco?.MjModel,
    hasMjDataClass: !!mujoco?.MjData,
    hasStaticMjLoadXml: typeof mujoco?.MjModel?.mj_loadXML === 'function',
    hasMjModelLoadFromXml: typeof mujoco?.MjModel?.loadFromXML === 'function',
    hasLegacyModelCtor: typeof mujoco?.Model === 'function',
    hasStateCtor: typeof mujoco?.State === 'function',
    hasSimulationCtor: typeof mujoco?.Simulation === 'function',
    hasMjStep: typeof mujoco?.mj_step === 'function',
    hasMjForward: typeof mujoco?.mj_forward === 'function',
    hasMjResetData: typeof mujoco?.mj_resetData === 'function',
  };
}

export function mountWorkingFilesystem(mujoco) {
  try {
    mujoco.FS.mkdir('/working');
  } catch {
    // Directory may already exist.
  }

  try {
    mujoco.FS.mount(mujoco.MEMFS, { root: '.' }, '/working');
  } catch {
    // Repeated mounts are safe to ignore.
  }
}

export function unmountWorkingFilesystem(mujoco) {
  try {
    mujoco.FS.unmount('/working');
  } catch {
    // ignore repeated unmounts
  }
}

export function loadModelFromXml({ mujoco, xmlPath, xmlText = null, log = () => {} }) {
  const apiSummary = summarizeMujocoApi(mujoco);

  if (typeof mujoco?.MjModel?.loadFromXML === 'function') {
    let parseOk = true;
    let parseError = '';

    if (xmlText && typeof DOMParser !== 'undefined') {
      try {
        const parsed = new DOMParser().parseFromString(xmlText, 'application/xml');
        const errorNode = parsed.querySelector('parsererror');

        if (errorNode) {
          parseOk = false;
          parseError = (errorNode.textContent || '').trim().slice(0, 400);
        }
      } catch (error) {
        parseOk = false;
        parseError = error instanceof Error ? error.message : String(error);
      }
    }

    if (!parseOk) {
      throw new Error(`Generated MJCF is not valid XML. ${parseError}`);
    }

    log(`Trying MjModel.loadFromXML(path): ${xmlPath}`);

    let model = null;

    try {
      model = mujoco.MjModel.loadFromXML(xmlPath);
    } catch (error) {
      log(`MjModel.loadFromXML(path) threw: ${error instanceof Error ? error.message : String(error)}`, 'warn');
    }

    if (model) return model;

    if (xmlText) {
      log('MjModel.loadFromXML(path) returned null, trying XML string fallback.', 'warn');

      try {
        model = mujoco.MjModel.loadFromXML(xmlText);
      } catch (error) {
        log(`MjModel.loadFromXML(xmlString) threw: ${error instanceof Error ? error.message : String(error)}`, 'warn');
      }

      if (model) return model;
    }

    throw new Error(
      `MjModel.loadFromXML returned null. path=${xmlPath} xmlLength=${xmlText ? xmlText.length : 0} ` +
        `MjModelKeys=${Object.keys(mujoco.MjModel).slice(0, 24).join(',')}`,
    );
  }

  if (typeof mujoco?.MjModel?.mj_loadXML === 'function') {
    log(`Trying MjModel.mj_loadXML: ${xmlPath}`);
    const model = mujoco.MjModel.mj_loadXML(xmlPath);
    if (!model) {
      throw new Error(`MjModel.mj_loadXML returned null. path=${xmlPath}`);
    }
    return model;
  }

  throw new Error(
    `MuJoCo API mismatch: no MjModel.loadFromXML / MjModel.mj_loadXML found. ` +
      `loadedKeys=${apiSummary.topLevelKeys.slice(0, 24).join(',')}`,
  );
}

export function createDataForModel(mujoco, model) {
  if (typeof mujoco?.MjData === 'function') {
    const data = new mujoco.MjData(model);
    if (!data) {
      throw new Error('Failed to create mjData.');
    }
    return data;
  }

  const apiSummary = summarizeMujocoApi(mujoco);
  throw new Error(
    `MuJoCo API mismatch: MjData constructor is missing. ` +
      `loadedKeys=${apiSummary.topLevelKeys.slice(0, 24).join(',')}`,
  );
}

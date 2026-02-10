/*!
 * OverlayScrollbars
 * Version: 2.14.0
 *
 * Copyright (c) Rene Haas | KingSora.
 * https://github.com/KingSora
 *
 * Released under the MIT license.
 */
var OverlayScrollbarsGlobal = function(t) {
  "use strict";
  const createCache = (t, n) => {
    const {o: o, i: s, u: e} = t;
    let c = o;
    let r;
    const cacheUpdateContextual = (t, n) => {
      const o = c;
      const l = t;
      const i = n || (s ? !s(o, l) : o !== l);
      if (i || e) {
        c = l;
        r = o;
      }
      return [ c, i, r ];
    };
    const cacheUpdateIsolated = t => cacheUpdateContextual(n(c, r), t);
    const getCurrentCache = t => [ c, !!t, r ];
    return [ n ? cacheUpdateIsolated : cacheUpdateContextual, getCurrentCache ];
  };
  const n = typeof window !== "undefined" && typeof HTMLElement !== "undefined" && !!window.document;
  const o = n ? window : {};
  const s = Math.max;
  const e = Math.min;
  const c = Math.round;
  const r = Math.abs;
  const l = Math.sign;
  const i = o.cancelAnimationFrame;
  const a = o.requestAnimationFrame;
  const u = o.setTimeout;
  const f = o.clearTimeout;
  const getApi = t => typeof o[t] !== "undefined" ? o[t] : void 0;
  const _ = getApi("MutationObserver");
  const d = getApi("IntersectionObserver");
  const p = getApi("ResizeObserver");
  const v = getApi("ScrollTimeline");
  const isUndefined = t => t === void 0;
  const isNull = t => t === null;
  const isNumber = t => typeof t === "number";
  const isString = t => typeof t === "string";
  const isBoolean = t => typeof t === "boolean";
  const isFunction = t => typeof t === "function";
  const isArray = t => Array.isArray(t);
  const isObject = t => typeof t === "object" && !isArray(t) && !isNull(t);
  const isArrayLike = t => {
    const n = !!t && t.length;
    const o = isNumber(n) && n > -1 && n % 1 == 0;
    return isArray(t) || !isFunction(t) && o ? n > 0 && isObject(t) ? n - 1 in t : true : false;
  };
  const isPlainObject = t => !!t && t.constructor === Object;
  const isHTMLElement = t => t instanceof HTMLElement;
  const isElement = t => t instanceof Element;
  const animationCurrentTime = () => performance.now();
  const animateNumber = (t, n, o, e, c) => {
    let r = 0;
    const l = animationCurrentTime();
    const u = s(0, o);
    const frame = o => {
      const i = animationCurrentTime();
      const f = i - l;
      const _ = f >= u;
      const d = o ? 1 : 1 - (s(0, l + u - i) / u || 0);
      const p = (n - t) * (isFunction(c) ? c(d, d * u, 0, 1, u) : d) + t;
      const v = _ || d === 1;
      if (e) {
        e(p, d, v);
      }
      r = v ? 0 : a((() => frame()));
    };
    frame();
    return t => {
      i(r);
      if (t) {
        frame(t);
      }
    };
  };
  function each(t, n) {
    if (isArrayLike(t)) {
      for (let o = 0; o < t.length; o++) {
        if (n(t[o], o, t) === false) {
          break;
        }
      }
    } else if (t) {
      each(Object.keys(t), (o => n(t[o], o, t)));
    }
    return t;
  }
  const inArray = (t, n) => t.indexOf(n) >= 0;
  const concat = (t, n) => t.concat(n);
  const push = (t, n, o) => {
    if (!isString(n) && isArrayLike(n)) {
      Array.prototype.push.apply(t, n);
    } else {
      t.push(n);
    }
    return t;
  };
  const from = t => Array.from(t || []);
  const createOrKeepArray = t => {
    if (isArray(t)) {
      return t;
    }
    return !isString(t) && isArrayLike(t) ? from(t) : [ t ];
  };
  const isEmptyArray = t => !!t && !t.length;
  const deduplicateArray = t => from(new Set(t));
  const runEachAndClear = (t, n, o) => {
    const runFn = t => t ? t.apply(void 0, n || []) : true;
    each(t, runFn);
    if (!o) {
      t.length = 0;
    }
  };
  const g = "paddingTop";
  const h = "paddingRight";
  const b = "paddingLeft";
  const w = "paddingBottom";
  const y = "marginLeft";
  const S = "marginRight";
  const m = "marginBottom";
  const O = "overflowX";
  const C = "overflowY";
  const $ = "width";
  const x = "height";
  const H = "visible";
  const E = "hidden";
  const z = "scroll";
  const capitalizeFirstLetter = t => {
    const n = String(t || "");
    return n ? n[0].toUpperCase() + n.slice(1) : "";
  };
  const equal = (t, n, o, s) => {
    if (t && n) {
      let s = true;
      each(o, (o => {
        const e = t[o];
        const c = n[o];
        if (e !== c) {
          s = false;
        }
      }));
      return s;
    }
    return false;
  };
  const equalWH = (t, n) => equal(t, n, [ "w", "h" ]);
  const equalXY = (t, n) => equal(t, n, [ "x", "y" ]);
  const equalTRBL = (t, n) => equal(t, n, [ "t", "r", "b", "l" ]);
  const bind = (t, ...n) => t.bind(0, ...n);
  const selfClearTimeout = t => {
    let n;
    const o = t ? u : a;
    const s = t ? f : i;
    return [ e => {
      s(n);
      n = o((() => e()), isFunction(t) ? t() : t);
    }, () => s(n) ];
  };
  const getDebouncer = t => {
    const n = isFunction(t) ? t() : t;
    if (isNumber(n)) {
      const t = n ? u : a;
      const o = n ? f : i;
      return s => {
        const e = t((() => s()), n);
        return () => {
          o(e);
        };
      };
    }
    return n && n._;
  };
  const debounce = (t, n) => {
    const {p: o, v: s, S: e, m: c} = n || {};
    let r;
    let l;
    let i;
    let a;
    const u = function invokeFunctionToDebounce(n) {
      if (l) {
        l();
      }
      if (r) {
        r();
      }
      a = l = r = i = void 0;
      t.apply(this, n);
    };
    const mergeParms = t => c && i ? c(i, t) : t;
    const flush = () => {
      if (l && i) {
        u(mergeParms(i) || i);
      }
    };
    const f = function debouncedFn() {
      const t = from(arguments);
      const n = getDebouncer(o);
      if (n) {
        const o = typeof e === "function" ? e() : e;
        const c = getDebouncer(s);
        const f = mergeParms(t);
        const _ = f || t;
        const d = u.bind(0, _);
        if (l) {
          l();
        }
        if (o && !a) {
          d();
          a = true;
          l = n((() => a = void 0));
        } else {
          l = n(d);
          if (c && !r) {
            r = c(flush);
          }
        }
        i = _;
      } else {
        u(t);
      }
    };
    f.O = flush;
    return f;
  };
  const hasOwnProperty = (t, n) => Object.prototype.hasOwnProperty.call(t, n);
  const keys = t => t ? Object.keys(t) : [];
  const assignDeep = (t, n, o, s, e, c, r) => {
    const l = [ n, o, s, e, c, r ];
    if ((typeof t !== "object" || isNull(t)) && !isFunction(t)) {
      t = {};
    }
    each(l, (n => {
      each(n, ((o, s) => {
        const e = n[s];
        if (t === e) {
          return true;
        }
        const c = isArray(e);
        if (e && isPlainObject(e)) {
          const n = t[s];
          let o = n;
          if (c && !isArray(n)) {
            o = [];
          } else if (!c && !isPlainObject(n)) {
            o = {};
          }
          t[s] = assignDeep(o, e);
        } else {
          t[s] = c ? e.slice() : e;
        }
      }));
    }));
    return t;
  };
  const removeUndefinedProperties = (t, n) => each(assignDeep({}, t), ((t, n, o) => {
    if (t === void 0) {
      delete o[n];
    } else if (t && isPlainObject(t)) {
      o[n] = removeUndefinedProperties(t);
    }
  }));
  const isEmptyObject = t => !keys(t).length;
  const noop = () => {};
  const capNumber = (t, n, o) => s(t, e(n, o));
  const getDomTokensArray = t => deduplicateArray((isArray(t) ? t : (t || "").split(" ")).filter((t => t)));
  const getAttr = (t, n) => t && t.getAttribute(n);
  const hasAttr = (t, n) => t && t.hasAttribute(n);
  const setAttrs = (t, n, o) => {
    each(getDomTokensArray(n), (n => {
      if (t) {
        t.setAttribute(n, String(o || ""));
      }
    }));
  };
  const removeAttrs = (t, n) => {
    each(getDomTokensArray(n), (n => t && t.removeAttribute(n)));
  };
  const domTokenListAttr = (t, n) => {
    const o = getDomTokensArray(getAttr(t, n));
    const s = bind(setAttrs, t, n);
    const domTokenListOperation = (t, n) => {
      const s = new Set(o);
      each(getDomTokensArray(t), (t => {
        s[n](t);
      }));
      return from(s).join(" ");
    };
    return {
      C: t => s(domTokenListOperation(t, "delete")),
      $: t => s(domTokenListOperation(t, "add")),
      H: t => {
        const n = getDomTokensArray(t);
        return n.reduce(((t, n) => t && o.includes(n)), n.length > 0);
      }
    };
  };
  const removeAttrClass = (t, n, o) => {
    domTokenListAttr(t, n).C(o);
    return bind(addAttrClass, t, n, o);
  };
  const addAttrClass = (t, n, o) => {
    domTokenListAttr(t, n).$(o);
    return bind(removeAttrClass, t, n, o);
  };
  const addRemoveAttrClass = (t, n, o, s) => (s ? addAttrClass : removeAttrClass)(t, n, o);
  const hasAttrClass = (t, n, o) => domTokenListAttr(t, n).H(o);
  const createDomTokenListClass = t => domTokenListAttr(t, "class");
  const removeClass = (t, n) => {
    createDomTokenListClass(t).C(n);
  };
  const addClass = (t, n) => {
    createDomTokenListClass(t).$(n);
    return bind(removeClass, t, n);
  };
  const find = (t, n) => {
    const o = n ? isElement(n) && n : document;
    return o ? from(o.querySelectorAll(t)) : [];
  };
  const findFirst = (t, n) => {
    const o = n ? isElement(n) && n : document;
    return o && o.querySelector(t);
  };
  const is = (t, n) => isElement(t) && t.matches(n);
  const isBodyElement = t => is(t, "body");
  const contents = t => t ? from(t.childNodes) : [];
  const parent = t => t && t.parentElement;
  const closest = (t, n) => isElement(t) && t.closest(n);
  const getFocusedElement = t => document.activeElement;
  const liesBetween = (t, n, o) => {
    const s = closest(t, n);
    const e = t && findFirst(o, s);
    const c = closest(e, n) === s;
    return s && e ? s === t || e === t || c && closest(closest(t, o), n) !== s : false;
  };
  const removeElements = t => {
    each(createOrKeepArray(t), (t => {
      const n = parent(t);
      if (t && n) {
        n.removeChild(t);
      }
    }));
  };
  const appendChildren = (t, n) => bind(removeElements, t && n && each(createOrKeepArray(n), (n => {
    if (n) {
      t.appendChild(n);
    }
  })));
  let I;
  const getTrustedTypePolicy = () => I;
  const setTrustedTypePolicy = t => {
    I = t;
  };
  const createDiv = t => {
    const n = document.createElement("div");
    setAttrs(n, "class", t);
    return n;
  };
  const createDOM = t => {
    const n = createDiv();
    const o = getTrustedTypePolicy();
    const s = t.trim();
    n.innerHTML = o ? o.createHTML(s) : s;
    return each(contents(n), (t => removeElements(t)));
  };
  const getCSSVal = (t, n) => t.getPropertyValue(n) || t[n] || "";
  const validFiniteNumber = t => {
    const n = t || 0;
    return isFinite(n) ? n : 0;
  };
  const parseToZeroOrNumber = t => validFiniteNumber(parseFloat(t || ""));
  const roundCssNumber = t => Math.round(t * 1e4) / 1e4;
  const numberToCssPx = t => `${roundCssNumber(validFiniteNumber(t))}px`;
  function setStyles(t, n) {
    t && n && each(n, ((n, o) => {
      try {
        const s = t.style;
        const e = isNull(n) || isBoolean(n) ? "" : isNumber(n) ? numberToCssPx(n) : n;
        if (o.indexOf("--") === 0) {
          s.setProperty(o, e);
        } else {
          s[o] = e;
        }
      } catch (s) {}
    }));
  }
  function getStyles(t, n, s) {
    const e = isString(n);
    let c = e ? "" : {};
    if (t) {
      const r = o.getComputedStyle(t, s) || t.style;
      c = e ? getCSSVal(r, n) : from(n).reduce(((t, n) => {
        t[n] = getCSSVal(r, n);
        return t;
      }), c);
    }
    return c;
  }
  const topRightBottomLeft = (t, n, o) => {
    const s = n ? `${n}-` : "";
    const e = o ? `-${o}` : "";
    const c = `${s}top${e}`;
    const r = `${s}right${e}`;
    const l = `${s}bottom${e}`;
    const i = `${s}left${e}`;
    const a = getStyles(t, [ c, r, l, i ]);
    return {
      t: parseToZeroOrNumber(a[c]),
      r: parseToZeroOrNumber(a[r]),
      b: parseToZeroOrNumber(a[l]),
      l: parseToZeroOrNumber(a[i])
    };
  };
  const getTrasformTranslateValue = (t, n) => `translate${isObject(t) ? `(${t.x},${t.y})` : `${n ? "X" : "Y"}(${t})`}`;
  const elementHasDimensions = t => !!(t.offsetWidth || t.offsetHeight || t.getClientRects().length);
  const A = {
    w: 0,
    h: 0
  };
  const getElmWidthHeightProperty = (t, n) => n ? {
    w: n[`${t}Width`],
    h: n[`${t}Height`]
  } : A;
  const getWindowSize = t => getElmWidthHeightProperty("inner", t || o);
  const T = bind(getElmWidthHeightProperty, "offset");
  const D = bind(getElmWidthHeightProperty, "client");
  const M = bind(getElmWidthHeightProperty, "scroll");
  const getFractionalSize = t => {
    const n = parseFloat(getStyles(t, $)) || 0;
    const o = parseFloat(getStyles(t, x)) || 0;
    return {
      w: n - c(n),
      h: o - c(o)
    };
  };
  const getBoundingClientRect = t => t.getBoundingClientRect();
  const hasDimensions = t => !!t && elementHasDimensions(t);
  const domRectHasDimensions = t => !!(t && (t[x] || t[$]));
  const domRectAppeared = (t, n) => {
    const o = domRectHasDimensions(t);
    const s = domRectHasDimensions(n);
    return !s && o;
  };
  const removeEventListener = (t, n, o, s) => {
    each(getDomTokensArray(n), (n => {
      if (t) {
        t.removeEventListener(n, o, s);
      }
    }));
  };
  const addEventListener = (t, n, o, s) => {
    var e;
    const c = (e = s && s.I) != null ? e : true;
    const r = s && s.A || false;
    const l = s && s.T || false;
    const i = {
      passive: c,
      capture: r
    };
    return bind(runEachAndClear, getDomTokensArray(n).map((n => {
      const s = l ? e => {
        removeEventListener(t, n, s, r);
        if (o) {
          o(e);
        }
      } : o;
      if (t) {
        t.addEventListener(n, s, i);
      }
      return bind(removeEventListener, t, n, s, r);
    })));
  };
  const stopPropagation = t => t.stopPropagation();
  const preventDefault = t => t.preventDefault();
  const stopAndPrevent = t => stopPropagation(t) || preventDefault(t);
  const scrollElementTo = (t, n) => {
    const {x: o, y: s} = isNumber(n) ? {
      x: n,
      y: n
    } : n || {};
    isNumber(o) && (t.scrollLeft = o);
    isNumber(s) && (t.scrollTop = s);
  };
  const getElementScroll = t => ({
    x: t.scrollLeft,
    y: t.scrollTop
  });
  const getZeroScrollCoordinates = () => ({
    D: {
      x: 0,
      y: 0
    },
    M: {
      x: 0,
      y: 0
    }
  });
  const sanitizeScrollCoordinates = (t, n) => {
    const {D: o, M: s} = t;
    const {w: e, h: c} = n;
    const sanitizeAxis = (t, n, o) => {
      let s = l(t) * o;
      let e = l(n) * o;
      if (s === e) {
        const o = r(t);
        const c = r(n);
        e = o > c ? 0 : e;
        s = o < c ? 0 : s;
      }
      s = s === e ? 0 : s;
      return [ s + 0, e + 0 ];
    };
    const [i, a] = sanitizeAxis(o.x, s.x, e);
    const [u, f] = sanitizeAxis(o.y, s.y, c);
    return {
      D: {
        x: i,
        y: u
      },
      M: {
        x: a,
        y: f
      }
    };
  };
  const isDefaultDirectionScrollCoordinates = ({D: t, M: n}) => {
    const getAxis = (t, n) => t === 0 && t <= n;
    return {
      x: getAxis(t.x, n.x),
      y: getAxis(t.y, n.y)
    };
  };
  const getScrollCoordinatesPercent = ({D: t, M: n}, o) => {
    const getAxis = (t, n, o) => capNumber(0, 1, (t - o) / (t - n) || 0);
    return {
      x: getAxis(t.x, n.x, o.x),
      y: getAxis(t.y, n.y, o.y)
    };
  };
  const focusElement = t => {
    if (t && t.focus) {
      t.focus({
        preventScroll: true,
        focusVisible: false
      });
    }
  };
  const manageListener = (t, n) => {
    each(createOrKeepArray(n), t);
  };
  const createEventListenerHub = t => {
    const n = new Map;
    const removeEvent = (t, o) => {
      if (t) {
        const s = n.get(t);
        manageListener((t => {
          if (s) {
            s[t ? "delete" : "clear"](t);
          }
        }), o);
      } else {
        n.forEach((t => {
          t.clear();
        }));
        n.clear();
      }
    };
    const addEvent = (t, o) => {
      if (isString(t)) {
        const s = n.get(t) || new Set;
        n.set(t, s);
        manageListener((t => {
          if (isFunction(t)) {
            s.add(t);
          }
        }), o);
        return bind(removeEvent, t, o);
      }
      if (isBoolean(o) && o) {
        removeEvent();
      }
      const s = keys(t);
      const e = [];
      each(s, (n => {
        const o = t[n];
        if (o) {
          push(e, addEvent(n, o));
        }
      }));
      return bind(runEachAndClear, e);
    };
    const triggerEvent = (t, o) => {
      each(from(n.get(t)), (t => {
        if (o && !isEmptyArray(o)) {
          t.apply(0, o);
        } else {
          t();
        }
      }));
    };
    addEvent(t || {});
    return [ addEvent, removeEvent, triggerEvent ];
  };
  const k = {};
  const R = {};
  const addPlugins = t => {
    each(t, (t => each(t, ((n, o) => {
      k[o] = t[o];
    }))));
  };
  const registerPluginModuleInstances = (t, n, o) => keys(t).map((s => {
    const {static: e, instance: c} = t[s];
    const [r, l, i] = o || [];
    const a = o ? c : e;
    if (a) {
      const t = o ? a(r, l, n) : a(n);
      return (i || R)[s] = t;
    }
  }));
  const getStaticPluginModuleInstance = t => R[t];
  const V = "__osOptionsValidationPlugin";
  const L = `data-overlayscrollbars`;
  const U = "os-environment";
  const P = `${U}-scrollbar-hidden`;
  const N = `${L}-initialize`;
  const q = "noClipping";
  const j = `${L}-body`;
  const B = L;
  const F = "host";
  const X = `${L}-viewport`;
  const Y = O;
  const W = C;
  const G = "arrange";
  const J = "measuring";
  const K = "scrolling";
  const Q = "scrollbarHidden";
  const Z = "noContent";
  const tt = `${L}-padding`;
  const nt = `${L}-content`;
  const ot = "os-size-observer";
  const st = `${ot}-appear`;
  const et = `${ot}-listener`;
  const ct = `${et}-scroll`;
  const rt = `${et}-item`;
  const lt = `${rt}-final`;
  const it = "os-trinsic-observer";
  const at = "os-theme-none";
  const ut = "os-scrollbar";
  const ft = `${ut}-rtl`;
  const _t = `${ut}-horizontal`;
  const dt = `${ut}-vertical`;
  const pt = `${ut}-track`;
  const vt = `${ut}-handle`;
  const gt = `${ut}-visible`;
  const ht = `${ut}-cornerless`;
  const bt = `${ut}-interaction`;
  const wt = `${ut}-unusable`;
  const yt = `${ut}-auto-hide`;
  const St = `${yt}-hidden`;
  const mt = `${ut}-wheel`;
  const Ot = `${pt}-interactive`;
  const Ct = `${vt}-interactive`;
  const $t = "__osSizeObserverPlugin";
  const xt = /* @__PURE__ */ (() => ({
    [$t]: {
      static: () => (t, n, o) => {
        const s = 3333333;
        const e = "scroll";
        const c = createDOM(`<div class="${rt}" dir="ltr"><div class="${rt}"><div class="${lt}"></div></div><div class="${rt}"><div class="${lt}" style="width: 200%; height: 200%"></div></div></div>`);
        const r = c[0];
        const l = r.lastChild;
        const u = r.firstChild;
        const f = u == null ? void 0 : u.firstChild;
        let _ = T(r);
        let d = _;
        let p = false;
        let v;
        const reset = () => {
          scrollElementTo(u, s);
          scrollElementTo(l, s);
        };
        const onResized = t => {
          v = 0;
          if (p) {
            _ = d;
            n(t === true);
          }
        };
        const onScroll = t => {
          d = T(r);
          p = !t || !equalWH(d, _);
          if (t) {
            stopPropagation(t);
            if (p && !v) {
              i(v);
              v = a(onResized);
            }
          } else {
            onResized(t === false);
          }
          reset();
        };
        const g = [ appendChildren(t, c), addEventListener(u, e, onScroll), addEventListener(l, e, onScroll) ];
        addClass(t, ct);
        setStyles(f, {
          [$]: s,
          [x]: s
        });
        a(reset);
        return [ o ? bind(onScroll, false) : reset, g ];
      }
    }
  }))();
  const getShowNativeOverlaidScrollbars = (t, n) => {
    const {k: o} = n;
    const [s, e] = t("showNativeOverlaidScrollbars");
    return [ s && o.x && o.y, e ];
  };
  const overflowIsVisible = t => t.indexOf(H) === 0;
  const overflowBehaviorToOverflowStyle = t => t.replace(`${H}-`, "");
  const overflowCssValueToOverflowStyle = (t, n) => {
    if (t === "auto") {
      return n ? z : E;
    }
    const o = t || E;
    return [ E, z, H ].includes(o) ? o : E;
  };
  const getElementOverflowStyle = (t, n) => {
    const {overflowX: o, overflowY: s} = getStyles(t, [ O, C ]);
    return {
      x: overflowCssValueToOverflowStyle(o, n.x),
      y: overflowCssValueToOverflowStyle(s, n.y)
    };
  };
  const Ht = "__osScrollbarsHidingPlugin";
  const Et = /* @__PURE__ */ (() => ({
    [Ht]: {
      static: () => ({
        R: (t, n, o, s, e) => {
          const {V: c, L: r} = t;
          const {U: l, k: i, P: a} = s;
          const u = !c && !l && (i.x || i.y);
          const [f] = getShowNativeOverlaidScrollbars(e, s);
          const _getViewportOverflowHideOffset = t => {
            const n = l || f ? 0 : 42;
            const getHideOffsetPerAxis = (t, o, s) => {
              const e = t ? n : s;
              const c = o && !l ? e : 0;
              const r = t && !!n;
              return [ c, r ];
            };
            const [o, s] = getHideOffsetPerAxis(i.x, t.x === z, a.x);
            const [e, c] = getHideOffsetPerAxis(i.y, t.y === z, a.y);
            return {
              N: {
                x: o,
                y: e
              },
              q: {
                x: s,
                y: c
              }
            };
          };
          const _hideNativeScrollbars = t => {
            if (!c) {
              const {j: s} = o;
              const e = assignDeep({}, {
                [S]: 0,
                [m]: 0,
                [y]: 0
              });
              const {N: c, q: r} = _getViewportOverflowHideOffset(t);
              const {x: l, y: i} = r;
              const {x: a, y: f} = c;
              const {B: _} = n;
              const d = s ? y : S;
              const p = s ? b : h;
              const v = _[d];
              const g = _[m];
              const O = _[p];
              const C = _[w];
              e[$] = `calc(100% + ${f + v * -1}px)`;
              e[d] = -f + v;
              e[m] = -a + g;
              if (u) {
                e[p] = O + (i ? f : 0);
                e[w] = C + (l ? a : 0);
              }
              return e;
            }
          };
          const _arrangeViewport = (t, s, e) => {
            if (u) {
              const {B: c} = n;
              const {N: l, q: i} = _getViewportOverflowHideOffset(t);
              const {x: a, y: u} = i;
              const {x: f, y: _} = l;
              const {j: d} = o;
              const p = d ? h : b;
              const v = c[p];
              const g = c.paddingTop;
              const w = s.w + e.w;
              const y = s.h + e.h;
              const S = {
                w: _ && u ? `${_ + w - v}px` : "",
                h: f && a ? `${f + y - g}px` : ""
              };
              setStyles(r, {
                "--os-vaw": S.w,
                "--os-vah": S.h
              });
            }
            return u;
          };
          const _undoViewportArrange = () => {
            if (u) {
              const {F: t, B: o} = n;
              const s = getElementOverflowStyle(r, t);
              const {q: e} = _getViewportOverflowHideOffset(s);
              const {x: c, y: l} = e;
              const i = {};
              const assignProps = t => each(t, (t => {
                i[t] = o[t];
              }));
              if (c) {
                assignProps([ m, g, w ]);
              }
              if (l) {
                assignProps([ y, S, b, h ]);
              }
              const a = getStyles(r, keys(i));
              const u = removeAttrClass(r, X, G);
              setStyles(r, i);
              return () => {
                setStyles(r, assignDeep({}, a, _hideNativeScrollbars(s)));
                u();
              };
            }
            return noop;
          };
          return {
            X: _arrangeViewport,
            Y: _undoViewportArrange,
            W: _hideNativeScrollbars
          };
        }
      })
    }
  }))();
  const zt = "__osClickScrollPlugin";
  const It = /* @__PURE__ */ (() => ({
    [zt]: {
      static: () => (t, n, o, s) => {
        let e = false;
        let c = noop;
        const r = 133;
        const l = 222;
        const [i, a] = selfClearTimeout(r);
        const u = Math.sign(n);
        const f = o * u;
        const _ = f / 2;
        const easing = t => 1 - (1 - t) * (1 - t);
        const easedEndPressAnimation = (n, o) => animateNumber(n, o, l, t, easing);
        const linearPressAnimation = (o, s) => animateNumber(o, n - f, r * s, ((o, s, e) => {
          t(o);
          if (e) {
            c = easedEndPressAnimation(o, n);
          }
        }));
        const d = animateNumber(0, f, l, ((r, l, a) => {
          t(r);
          if (a) {
            s(e);
            if (!e) {
              const t = n - r;
              const s = Math.sign(t - _) === u;
              if (s) {
                i((() => {
                  const s = t - f;
                  const e = Math.sign(s) === u;
                  c = e ? linearPressAnimation(r, Math.abs(s) / o) : easedEndPressAnimation(r, n);
                }));
              }
            }
          }
        }), easing);
        return t => {
          e = true;
          if (t) {
            d();
          }
          a();
          c();
        };
      }
    }
  }))();
  const opsStringify = t => JSON.stringify(t, ((t, n) => {
    if (isFunction(n)) {
      throw 0;
    }
    return n;
  }));
  const getPropByPath = (t, n) => t ? `${n}`.split(".").reduce(((t, n) => t && hasOwnProperty(t, n) ? t[n] : void 0), t) : void 0;
  const At = [ 0, 33 ];
  const Tt = [ 33, 99 ];
  const Dt = [ 222, 666, true ];
  const Mt = {
    paddingAbsolute: false,
    showNativeOverlaidScrollbars: false,
    update: {
      elementEvents: [ [ "img", "load" ] ],
      debounce: {
        mutation: At,
        resize: null,
        event: Tt,
        env: Dt
      },
      attributes: null,
      ignoreMutation: null,
      flowDirectionStyles: null
    },
    overflow: {
      x: "scroll",
      y: "scroll"
    },
    scrollbars: {
      theme: "os-theme-dark",
      visibility: "auto",
      autoHide: "never",
      autoHideDelay: 1300,
      autoHideSuspend: false,
      dragScroll: true,
      clickScroll: false,
      pointers: [ "mouse", "touch", "pen" ]
    }
  };
  const getOptionsDiff = (t, n) => {
    const o = {};
    const s = concat(keys(n), keys(t));
    each(s, (s => {
      const e = t[s];
      const c = n[s];
      if (isObject(e) && isObject(c)) {
        assignDeep(o[s] = {}, getOptionsDiff(e, c));
        if (isEmptyObject(o[s])) {
          delete o[s];
        }
      } else if (hasOwnProperty(n, s) && c !== e) {
        let t = true;
        if (isArray(e) || isArray(c)) {
          try {
            if (opsStringify(e) === opsStringify(c)) {
              t = false;
            }
          } catch (r) {}
        }
        if (t) {
          o[s] = c;
        }
      }
    }));
    return o;
  };
  const createOptionCheck = (t, n, o) => s => [ getPropByPath(t, s), o || getPropByPath(n, s) !== void 0 ];
  let kt;
  const getNonce = () => kt;
  const setNonce = t => {
    kt = t;
  };
  let Rt;
  const createEnvironment = () => {
    const getNativeScrollbarSize = (t, n, o) => {
      appendChildren(document.body, t);
      appendChildren(document.body, t);
      const s = D(t);
      const e = T(t);
      const c = getFractionalSize(n);
      if (o) {
        removeElements(t);
      }
      return {
        x: e.h - s.h + c.h,
        y: e.w - s.w + c.w
      };
    };
    const getNativeScrollbarsHiding = t => {
      let n = false;
      const o = addClass(t, P);
      try {
        n = getStyles(t, "scrollbar-width") === "none" || getStyles(t, "display", "::-webkit-scrollbar") === "none";
      } catch (s) {}
      o();
      return n;
    };
    const t = `.${U}{scroll-behavior:auto!important;position:fixed;opacity:0;visibility:hidden;overflow:scroll;height:200px;width:200px;z-index:-1}.${U} div{width:200%;height:200%;margin:10px 0}.${P}{scrollbar-width:none!important}.${P}::-webkit-scrollbar,.${P}::-webkit-scrollbar-corner{appearance:none!important;display:none!important;width:0!important;height:0!important}`;
    const n = createDOM(`<div class="${U}"><div></div><style>${t}</style></div>`);
    const s = n[0];
    const e = s.firstChild;
    const c = s.lastChild;
    const r = getNonce();
    if (r) {
      c.nonce = r;
    }
    const [l, , i] = createEventListenerHub();
    const [a, u] = createCache({
      o: getNativeScrollbarSize(s, e),
      i: equalXY
    }, bind(getNativeScrollbarSize, s, e, true));
    const [f] = u();
    const _ = getNativeScrollbarsHiding(s);
    const d = {
      x: f.x === 0,
      y: f.y === 0
    };
    const p = {
      elements: {
        host: null,
        padding: !_,
        viewport: t => _ && isBodyElement(t) && t,
        content: false
      },
      scrollbars: {
        slot: true
      },
      cancel: {
        nativeScrollbarsOverlaid: false,
        body: null
      }
    };
    const g = assignDeep({}, Mt);
    const h = bind(assignDeep, {}, g);
    const b = bind(assignDeep, {}, p);
    const w = {
      P: f,
      k: d,
      U: _,
      G: !!v,
      J: bind(l, "r"),
      K: b,
      Z: t => assignDeep(p, t) && b(),
      tt: h,
      nt: t => assignDeep(g, t) && h(),
      ot: assignDeep({}, p),
      st: assignDeep({}, g)
    };
    removeAttrs(s, "style");
    removeElements(s);
    addEventListener(o, "resize", (() => {
      i("r", []);
    }));
    if (isFunction(o.matchMedia) && !_ && (!d.x || !d.y)) {
      const addZoomListener = t => {
        const n = o.matchMedia(`(resolution: ${o.devicePixelRatio}dppx)`);
        addEventListener(n, "change", (() => {
          t();
          addZoomListener(t);
        }), {
          T: true
        });
      };
      addZoomListener((() => {
        const [t, n] = a();
        assignDeep(w.P, t);
        i("r", [ n ]);
      }));
    }
    return w;
  };
  const getEnvironment = () => {
    if (!Rt) {
      Rt = createEnvironment();
    }
    return Rt;
  };
  const createEventContentChange = (t, n, o) => {
    let s = false;
    const e = o ? new WeakMap : false;
    const destroy = () => {
      s = true;
    };
    const updateElements = c => {
      if (e && o) {
        const r = o.map((n => {
          const [o, s] = n || [];
          const e = s && o ? (c || find)(o, t) : [];
          return [ e, s ];
        }));
        each(r, (o => each(o[0], (c => {
          const r = o[1];
          const l = e.get(c) || [];
          const i = t.contains(c);
          if (i && r) {
            const t = addEventListener(c, r, (o => {
              if (s) {
                t();
                e.delete(c);
              } else {
                n(o);
              }
            }));
            e.set(c, push(l, t));
          } else {
            runEachAndClear(l);
            e.delete(c);
          }
        }))));
      }
    };
    updateElements();
    return [ destroy, updateElements ];
  };
  const createDOMObserver = (t, n, o, s) => {
    let e = false;
    const {et: c, ct: r, rt: l, lt: i, it: a, ut: u} = s || {};
    const [f, d] = createEventContentChange(t, (() => e && o(true)), l);
    const p = c || [];
    const v = r || [];
    const g = concat(p, v);
    const observerCallback = (e, c) => {
      if (!isEmptyArray(c)) {
        const r = a || noop;
        const l = u || noop;
        const f = [];
        const _ = [];
        let p = false;
        let g = false;
        each(c, (o => {
          const {attributeName: e, target: c, type: a, oldValue: u, addedNodes: d, removedNodes: h} = o;
          const b = a === "attributes";
          const w = a === "childList";
          const y = t === c;
          const S = b && e;
          const m = S && getAttr(c, e || "");
          const O = isString(m) ? m : null;
          const C = S && u !== O;
          const $ = inArray(v, e) && C;
          if (n && (w || !y)) {
            const n = b && C;
            const a = n && i && is(c, i);
            const _ = a ? !r(c, e, u, O) : !b || n;
            const p = _ && !l(o, !!a, t, s);
            each(d, (t => push(f, t)));
            each(h, (t => push(f, t)));
            g = g || p;
          }
          if (!n && y && C && !r(c, e, u, O)) {
            push(_, e);
            p = p || $;
          }
        }));
        d((t => deduplicateArray(f).reduce(((n, o) => {
          push(n, find(t, o));
          return is(o, t) ? push(n, o) : n;
        }), [])));
        if (n) {
          if (!e && g) {
            o(false);
          }
          return [ false ];
        }
        if (!isEmptyArray(_) || p) {
          const t = [ deduplicateArray(_), p ];
          if (!e) {
            o.apply(0, t);
          }
          return t;
        }
      }
    };
    const h = new _(bind(observerCallback, false));
    return [ () => {
      h.observe(t, {
        attributes: true,
        attributeOldValue: true,
        attributeFilter: g,
        subtree: n,
        childList: n,
        characterData: n
      });
      e = true;
      return () => {
        if (e) {
          f();
          h.disconnect();
          e = false;
        }
      };
    }, () => {
      if (e) {
        return observerCallback(true, h.takeRecords());
      }
    } ];
  };
  let Vt = null;
  const createSizeObserver = (t, n, o) => {
    const {ft: s} = o || {};
    const e = getStaticPluginModuleInstance($t);
    const [c] = createCache({
      o: false,
      u: true
    });
    return () => {
      const o = [];
      const r = createDOM(`<div class="${ot}"><div class="${et}"></div></div>`);
      const l = r[0];
      const i = l.firstChild;
      const onSizeChangedCallbackProxy = t => {
        const o = isArray(t) && !isEmptyArray(t);
        let s = false;
        let e = false;
        if (o) {
          const n = t[0];
          const [o, , r] = c(n.contentRect);
          const l = domRectHasDimensions(o);
          e = domRectAppeared(o, r);
          s = !e && !l;
        } else {
          e = t === true;
        }
        if (!s) {
          n({
            _t: true,
            ft: e
          });
        }
      };
      if (p) {
        if (!isBoolean(Vt)) {
          const n = new p(noop);
          n.observe(t, {
            get box() {
              Vt = true;
            }
          });
          Vt = Vt || false;
          n.disconnect();
        }
        const n = debounce(onSizeChangedCallbackProxy, {
          p: 0,
          v: 0
        });
        const resizeObserverCallback = t => n(t);
        const s = new p(resizeObserverCallback);
        s.observe(Vt ? t : i);
        push(o, [ () => {
          s.disconnect();
        }, !Vt && appendChildren(t, l) ]);
        if (Vt) {
          const n = new p(resizeObserverCallback);
          n.observe(t, {
            box: "border-box"
          });
          push(o, (() => n.disconnect()));
        }
      } else if (e) {
        const [n, c] = e(i, onSizeChangedCallbackProxy, s);
        push(o, concat([ addClass(l, st), addEventListener(l, "animationstart", n), appendChildren(t, l) ], c));
      } else {
        return noop;
      }
      return bind(runEachAndClear, o);
    };
  };
  const createTrinsicObserver = (t, n) => {
    let o;
    const isHeightIntrinsic = t => t.h === 0 || t.isIntersecting || t.intersectionRatio > 0;
    const s = createDiv(it);
    const [e] = createCache({
      o: false
    });
    const triggerOnTrinsicChangedCallback = (t, o) => {
      if (t) {
        const s = e(isHeightIntrinsic(t));
        const [, c] = s;
        return c && !o && n(s) && [ s ];
      }
    };
    const intersectionObserverCallback = (t, n) => triggerOnTrinsicChangedCallback(n.pop(), t);
    return [ () => {
      const n = [];
      if (d) {
        o = new d(bind(intersectionObserverCallback, false), {
          root: t
        });
        o.observe(s);
        push(n, (() => {
          o.disconnect();
        }));
      } else {
        const onSizeChanged = () => {
          const t = T(s);
          triggerOnTrinsicChangedCallback(t);
        };
        push(n, createSizeObserver(s, onSizeChanged)());
        onSizeChanged();
      }
      return bind(runEachAndClear, push(n, appendChildren(t, s)));
    }, () => o && intersectionObserverCallback(true, o.takeRecords()) ];
  };
  const createObserversSetup = (t, n, o, s) => {
    let e;
    let c;
    let r;
    let l;
    let i;
    let a;
    let u;
    let f;
    const _ = `[${B}]`;
    const d = `[${X}]`;
    const v = [ "id", "class", "style", "open", "wrap", "cols", "rows" ];
    const {dt: g, vt: h, L: b, gt: w, ht: y, V: S, bt: m, wt: O, yt: C, St: $} = t;
    const getDirectionIsRTL = t => getStyles(t, "direction") === "rtl";
    const createDebouncedObservesUpdate = () => {
      let t;
      let n;
      let o;
      const e = debounce(s, {
        p: () => t,
        v: () => n,
        S: () => o,
        m(t, n) {
          const [o] = t;
          const [s] = n;
          return [ concat(keys(o), keys(s)).reduce(((t, n) => {
            t[n] = o[n] || s[n];
            return t;
          }), {}) ];
        }
      });
      const fn = (s, c) => {
        if (isArray(c)) {
          const [s, e, r] = c;
          t = s;
          n = e;
          o = r;
        } else if (isNumber(c)) {
          t = c;
          n = false;
          o = false;
        } else {
          t = false;
          n = false;
          o = false;
        }
        e(s);
      };
      fn.O = e.O;
      return fn;
    };
    const x = {
      Ot: false,
      j: getDirectionIsRTL(g)
    };
    const H = getEnvironment();
    const E = getStaticPluginModuleInstance(Ht);
    const [z] = createCache({
      i: equalWH,
      o: {
        w: 0,
        h: 0
      }
    }, (() => {
      const s = E && E.R(t, n, x, H, o).Y;
      const e = m && S;
      const c = !e && hasAttrClass(h, B, q);
      const r = !S && O(G);
      const l = r && getElementScroll(w);
      const i = l && $();
      const a = C(J, c);
      const u = r && s && s();
      const f = M(b);
      const _ = getFractionalSize(b);
      if (u) {
        u();
      }
      scrollElementTo(w, l);
      if (i) {
        i();
      }
      if (c) {
        a();
      }
      return {
        w: f.w + _.w,
        h: f.h + _.h
      };
    }));
    const I = createDebouncedObservesUpdate();
    const setDirection = t => {
      const n = getDirectionIsRTL(g);
      assignDeep(t, {
        Ct: f !== n
      });
      assignDeep(x, {
        j: n
      });
      f = n;
    };
    const onTrinsicChanged = (t, n) => {
      const [o, e] = t;
      const c = {
        $t: e
      };
      assignDeep(x, {
        Ot: o
      });
      if (!n) {
        s(c);
      }
      return c;
    };
    const onSizeChanged = ({_t: t, ft: n}) => {
      const o = n ? s : I;
      const e = {
        _t: t || n,
        ft: n
      };
      setDirection(e);
      o(e, c);
    };
    const onContentMutation = (t, n) => {
      const [, o] = z();
      const s = {
        xt: o
      };
      setDirection(s);
      if (o && !n) {
        I(s, t ? r : e);
      }
      return s;
    };
    const onHostMutation = (t, n, o) => {
      const s = {
        Ht: n
      };
      setDirection(s);
      if (n && !o) {
        I(s, e);
      }
      return s;
    };
    const [A, T] = y ? createTrinsicObserver(h, onTrinsicChanged) : [];
    const D = !S && createSizeObserver(h, onSizeChanged, {
      ft: true
    });
    const [k, R] = createDOMObserver(h, false, onHostMutation, {
      ct: v,
      et: v
    });
    const V = S && p && new p((t => {
      const n = t[t.length - 1].contentRect;
      onSizeChanged({
        _t: true,
        ft: domRectAppeared(n, u)
      });
      u = n;
    }));
    return [ () => {
      if (V) {
        V.observe(h);
      }
      const t = D && D();
      const n = A && A();
      const o = k();
      const s = H.J((t => {
        const [, n] = z();
        I({
          Et: t,
          xt: n,
          _t: m
        }, l);
      }));
      return () => {
        if (V) {
          V.disconnect();
        }
        if (t) {
          t();
        }
        if (n) {
          n();
        }
        if (a) {
          a();
        }
        o();
        s();
      };
    }, ({zt: t, It: n, At: o}) => {
      const s = {};
      const [u] = t("update.ignoreMutation");
      const [f, p] = t("update.attributes");
      const [g, h] = t("update.elementEvents");
      const [w, m] = t("update.debounce");
      const O = h || p;
      const C = n || o;
      const ignoreMutationFromOptions = t => isFunction(u) && u(t);
      if (O) {
        if (i) {
          i();
        }
        if (a) {
          a();
        }
        const [t, n] = createDOMObserver(y || b, true, onContentMutation, {
          et: concat(v, f || []),
          rt: g,
          lt: _,
          ut: (t, n) => {
            const {target: o, attributeName: s} = t;
            const e = !n && s && !S ? liesBetween(o, _, d) : false;
            return e || !!closest(o, `.${ut}`) || !!ignoreMutationFromOptions(t);
          }
        });
        a = t();
        i = n;
      }
      if (m) {
        I.O();
        if (isArray(w) || isNumber(w)) {
          e = w;
          c = false;
          r = Tt;
          l = Dt;
        } else if (isPlainObject(w)) {
          e = w.mutation;
          c = w.resize;
          r = w.event;
          l = w.env;
        } else {
          e = false;
          c = false;
          r = false;
          l = false;
        }
      }
      if (C) {
        const t = R();
        const n = T && T();
        const o = i && i();
        if (t) {
          assignDeep(s, onHostMutation(t[0], t[1], C));
        }
        if (n) {
          assignDeep(s, onTrinsicChanged(n[0], C));
        }
        if (o) {
          assignDeep(s, onContentMutation(o[0], C));
        }
      }
      setDirection(s);
      return s;
    }, x ];
  };
  const resolveInitialization = (t, n) => isFunction(n) ? n.apply(0, t) : n;
  const staticInitializationElement = (t, n, o, s) => {
    const e = isUndefined(s) ? o : s;
    const c = resolveInitialization(t, e);
    return c || n.apply(0, t);
  };
  const dynamicInitializationElement = (t, n, o, s) => {
    const e = isUndefined(s) ? o : s;
    const c = resolveInitialization(t, e);
    return !!c && (isHTMLElement(c) ? c : n.apply(0, t));
  };
  const cancelInitialization = (t, n) => {
    const {nativeScrollbarsOverlaid: o, body: s} = n || {};
    const {k: e, U: c, K: r} = getEnvironment();
    const {nativeScrollbarsOverlaid: l, body: i} = r().cancel;
    const a = o != null ? o : l;
    const u = isUndefined(s) ? i : s;
    const f = (e.x || e.y) && a;
    const _ = t && (isNull(u) ? !c : u);
    return !!f || !!_;
  };
  const createScrollbarsSetupElements = (t, n, o, s) => {
    const e = "--os-viewport-percent";
    const c = "--os-scroll-percent";
    const r = "--os-scroll-direction";
    const {K: l} = getEnvironment();
    const {scrollbars: i} = l();
    const {slot: a} = i;
    const {dt: u, vt: f, L: _, Tt: d, gt: p, bt: g, V: h} = n;
    const {scrollbars: b} = d ? {} : t;
    const {slot: w} = b || {};
    const y = [];
    const S = [];
    const m = [];
    const O = dynamicInitializationElement([ u, f, _ ], (() => h && g ? u : f), a, w);
    const initScrollTimeline = t => {
      if (v) {
        let n = null;
        let s = [];
        const e = new v({
          source: p,
          axis: t
        });
        const cancelAnimation = () => {
          if (n) {
            n.cancel();
          }
          n = null;
        };
        const _setScrollPercentAnimation = c => {
          const {Dt: r} = o;
          const l = isDefaultDirectionScrollCoordinates(r)[t];
          const i = t === "x";
          const a = [ getTrasformTranslateValue(0, i), getTrasformTranslateValue(`calc(-100% + 100cq${i ? "w" : "h"})`, i) ];
          const u = l ? a : a.reverse();
          if (s[0] === u[0] && s[1] === u[1]) {
            return cancelAnimation;
          }
          s = u;
          cancelAnimation();
          n = c.Mt.animate({
            clear: [ "left" ],
            transform: u
          }, {
            timeline: e
          });
          return cancelAnimation;
        };
        return {
          kt: _setScrollPercentAnimation
        };
      }
    };
    const C = {
      x: initScrollTimeline("x"),
      y: initScrollTimeline("y")
    };
    const getViewportPercent = () => {
      const {Rt: t, Vt: n} = o;
      const getAxisValue = (t, n) => capNumber(0, 1, t / (t + n) || 0);
      return {
        x: getAxisValue(n.x, t.x),
        y: getAxisValue(n.y, t.y)
      };
    };
    const scrollbarStructureAddRemoveClass = (t, n, o) => {
      const s = o ? addClass : removeClass;
      each(t, (t => {
        s(t.Lt, n);
      }));
    };
    const scrollbarStyle = (t, n) => {
      each(t, (t => {
        const [o, s] = n(t);
        setStyles(o, s);
      }));
    };
    const scrollbarsAddRemoveClass = (t, n, o) => {
      const s = isBoolean(o);
      const e = s ? o : true;
      const c = s ? !o : true;
      if (e) {
        scrollbarStructureAddRemoveClass(S, t, n);
      }
      if (c) {
        scrollbarStructureAddRemoveClass(m, t, n);
      }
    };
    const refreshScrollbarsHandleLength = () => {
      const t = getViewportPercent();
      const createScrollbarStyleFn = t => n => [ n.Lt, {
        [e]: roundCssNumber(t) + ""
      } ];
      scrollbarStyle(S, createScrollbarStyleFn(t.x));
      scrollbarStyle(m, createScrollbarStyleFn(t.y));
    };
    const refreshScrollbarsHandleOffset = () => {
      if (!v) {
        const {Dt: t} = o;
        const n = getScrollCoordinatesPercent(t, getElementScroll(p));
        const createScrollbarStyleFn = t => n => [ n.Lt, {
          [c]: roundCssNumber(t) + ""
        } ];
        scrollbarStyle(S, createScrollbarStyleFn(n.x));
        scrollbarStyle(m, createScrollbarStyleFn(n.y));
      }
    };
    const refreshScrollbarsScrollCoordinates = () => {
      const {Dt: t} = o;
      const n = isDefaultDirectionScrollCoordinates(t);
      const createScrollbarStyleFn = t => n => [ n.Lt, {
        [r]: t ? "0" : "1"
      } ];
      scrollbarStyle(S, createScrollbarStyleFn(n.x));
      scrollbarStyle(m, createScrollbarStyleFn(n.y));
      if (v) {
        S.forEach(C.x.kt);
        m.forEach(C.y.kt);
      }
    };
    const refreshScrollbarsScrollbarOffset = () => {
      if (h && !g) {
        const {Rt: t, Dt: n} = o;
        const s = isDefaultDirectionScrollCoordinates(n);
        const e = getScrollCoordinatesPercent(n, getElementScroll(p));
        const styleScrollbarPosition = n => {
          const {Lt: o} = n;
          const c = parent(o) === _ && o;
          const getTranslateValue = (t, n, o) => {
            const s = n * t;
            return numberToCssPx(o ? s : -s);
          };
          return [ c, c && {
            transform: getTrasformTranslateValue({
              x: getTranslateValue(e.x, t.x, s.x),
              y: getTranslateValue(e.y, t.y, s.y)
            })
          } ];
        };
        scrollbarStyle(S, styleScrollbarPosition);
        scrollbarStyle(m, styleScrollbarPosition);
      }
    };
    const generateScrollbarDOM = t => {
      const n = t ? "x" : "y";
      const o = t ? _t : dt;
      const e = createDiv(`${ut} ${o}`);
      const c = createDiv(pt);
      const r = createDiv(vt);
      const l = {
        Lt: e,
        Ut: c,
        Mt: r
      };
      const i = C[n];
      push(t ? S : m, l);
      push(y, [ appendChildren(e, c), appendChildren(c, r), bind(removeElements, e), i && i.kt(l), s(l, scrollbarsAddRemoveClass, t) ]);
      return l;
    };
    const $ = bind(generateScrollbarDOM, true);
    const x = bind(generateScrollbarDOM, false);
    const appendElements = () => {
      appendChildren(O, S[0].Lt);
      appendChildren(O, m[0].Lt);
      return bind(runEachAndClear, y);
    };
    $();
    x();
    return [ {
      Pt: refreshScrollbarsHandleLength,
      Nt: refreshScrollbarsHandleOffset,
      qt: refreshScrollbarsScrollCoordinates,
      jt: refreshScrollbarsScrollbarOffset,
      Bt: scrollbarsAddRemoveClass,
      Ft: {
        Xt: S,
        Yt: $,
        Wt: bind(scrollbarStyle, S)
      },
      Gt: {
        Xt: m,
        Yt: x,
        Wt: bind(scrollbarStyle, m)
      }
    }, appendElements ];
  };
  const createScrollbarsSetupEvents = (t, n, o, s) => (e, l, i) => {
    const {vt: a, L: f, V: _, gt: d, Jt: p, St: v} = n;
    const {Lt: g, Ut: h, Mt: b} = e;
    const [w, y] = selfClearTimeout(333);
    const [S, m] = selfClearTimeout(444);
    const scrollOffsetElementScrollBy = t => {
      if (isFunction(d.scrollBy)) {
        d.scrollBy({
          behavior: "smooth",
          left: t.x,
          top: t.y
        });
      }
    };
    const createInteractiveScrollEvents = () => {
      const n = "pointerup pointercancel lostpointercapture";
      const s = `client${i ? "X" : "Y"}`;
      const e = i ? $ : x;
      const l = i ? "left" : "top";
      const a = i ? "w" : "h";
      const u = i ? "x" : "y";
      const createRelativeHandleMove = (t, n) => s => {
        const {Rt: e} = o;
        const c = T(h)[a] - T(b)[a];
        const r = n * s / c;
        const l = r * e[u];
        scrollElementTo(d, {
          [u]: t + l
        });
      };
      const f = [];
      return addEventListener(h, "pointerdown", (o => {
        const i = closest(o.target, `.${vt}`) === b;
        const _ = i ? b : h;
        const g = t.scrollbars;
        const w = g[i ? "dragScroll" : "clickScroll"];
        const {button: y, isPrimary: O, pointerType: C} = o;
        const {pointers: $} = g;
        const x = y === 0 && O && w && ($ || []).includes(C);
        if (x) {
          runEachAndClear(f);
          m();
          const t = !i && (o.shiftKey || w === "instant");
          const g = bind(getBoundingClientRect, b);
          const y = bind(getBoundingClientRect, h);
          const getHandleOffset = (t, n) => (t || g())[l] - (n || y())[l];
          const O = c(getBoundingClientRect(d)[e]) / T(d)[a] || 1;
          const C = createRelativeHandleMove(getElementScroll(d)[u], 1 / O);
          const $ = o[s];
          const x = g();
          const H = y();
          const E = x[e];
          const z = getHandleOffset(x, H) + E / 2;
          const I = $ - H[l];
          const A = i ? 0 : I - z;
          const releasePointerCapture = t => {
            runEachAndClear(k);
            _.releasePointerCapture(t.pointerId);
          };
          const D = i || t;
          const M = v();
          const k = [ addEventListener(p, n, releasePointerCapture), addEventListener(p, "selectstart", (t => preventDefault(t)), {
            I: false
          }), addEventListener(h, n, releasePointerCapture), D && addEventListener(h, "pointermove", (t => C(A + (t[s] - $)))), D && (() => {
            const t = getElementScroll(d);
            M();
            const n = getElementScroll(d);
            const o = {
              x: n.x - t.x,
              y: n.y - t.y
            };
            if (r(o.x) > 3 || r(o.y) > 3) {
              v();
              scrollElementTo(d, t);
              scrollOffsetElementScrollBy(o);
              S(M);
            }
          }) ];
          _.setPointerCapture(o.pointerId);
          if (t) {
            C(A);
          } else if (!i) {
            const t = getStaticPluginModuleInstance(zt);
            if (t) {
              const n = t(C, A, E, (t => {
                if (t) {
                  M();
                } else {
                  push(k, M);
                }
              }));
              push(k, n);
              push(f, bind(n, true));
            }
          }
        }
      }));
    };
    let O = true;
    return bind(runEachAndClear, [ addEventListener(b, "pointermove pointerleave", s), addEventListener(g, "pointerenter", (() => {
      l(bt, true);
    })), addEventListener(g, "pointerleave pointercancel", (() => {
      l(bt, false);
    })), !_ && addEventListener(g, "mousedown", (() => {
      const t = getFocusedElement();
      if (hasAttr(t, X) || hasAttr(t, B) || t === document.body) {
        u(bind(focusElement, f), 25);
      }
    })), addEventListener(g, "wheel", (t => {
      const {deltaX: n, deltaY: o, deltaMode: s} = t;
      if (O && s === 0 && parent(g) === a) {
        scrollOffsetElementScrollBy({
          x: n,
          y: o
        });
      }
      O = false;
      l(mt, true);
      w((() => {
        O = true;
        l(mt);
      }));
      preventDefault(t);
    }), {
      I: false,
      A: true
    }), addEventListener(g, "pointerdown", (() => {
      const t = addEventListener(p, "click", (t => {
        n();
        stopAndPrevent(t);
      }), {
        T: true,
        A: true,
        I: false
      });
      const n = addEventListener(p, "pointerup pointercancel", (() => {
        n();
        setTimeout(t, 150);
      }), {
        A: true,
        I: true
      });
    }), {
      A: true,
      I: true
    }), createInteractiveScrollEvents(), y, m ]);
  };
  const createScrollbarsSetup = (t, n, o, s, e, c) => {
    let r;
    let l;
    let i;
    let a;
    let u;
    let f = noop;
    let _ = 0;
    const d = [ "mouse", "pen" ];
    const isHoverablePointerType = t => d.includes(t.pointerType);
    const [p, v] = selfClearTimeout();
    const [g, h] = selfClearTimeout(100);
    const [b, w] = selfClearTimeout(100);
    const [y, S] = selfClearTimeout((() => _));
    const [m, O] = createScrollbarsSetupElements(t, e, s, createScrollbarsSetupEvents(n, e, s, (t => isHoverablePointerType(t) && manageScrollbarsAutoHideInstantInteraction())));
    const {vt: C, Kt: $, bt: x} = e;
    const {Bt: E, Pt: I, Nt: A, qt: T, jt: D} = m;
    const manageScrollbarsAutoHide = (t, n) => {
      S();
      if (t) {
        E(St);
      } else {
        const t = bind(E, St, true);
        if (_ > 0 && !n) {
          y(t);
        } else {
          t();
        }
      }
    };
    const manageScrollbarsAutoHideInstantInteraction = () => {
      if (i ? !r : !a) {
        manageScrollbarsAutoHide(true);
        g((() => {
          manageScrollbarsAutoHide(false);
        }));
      }
    };
    const manageAutoHideSuspension = t => {
      E(yt, t, true);
      E(yt, t, false);
    };
    const onHostMouseEnter = t => {
      if (isHoverablePointerType(t)) {
        r = i;
        if (i) {
          manageScrollbarsAutoHide(true);
        }
      }
    };
    const M = [ S, h, w, v, () => f(), addEventListener(C, "pointerover", onHostMouseEnter, {
      T: true
    }), addEventListener(C, "pointerenter", onHostMouseEnter), addEventListener(C, "pointerleave", (t => {
      if (isHoverablePointerType(t)) {
        r = false;
        if (i) {
          manageScrollbarsAutoHide(false);
        }
      }
    })), addEventListener(C, "pointermove", (t => {
      if (isHoverablePointerType(t) && l) {
        manageScrollbarsAutoHideInstantInteraction();
      }
    })), addEventListener($, "scroll", (t => {
      p((() => {
        A();
        manageScrollbarsAutoHideInstantInteraction();
      }));
      c(t);
      D();
    })) ];
    const k = getStaticPluginModuleInstance(Ht);
    return [ () => bind(runEachAndClear, push(M, O())), ({zt: t, At: n, Qt: e, Zt: c}) => {
      const {tn: r, nn: d, sn: p, en: v} = c || {};
      const {Ct: g, ft: h} = e || {};
      const {j: w} = o;
      const {k: y, U: S} = getEnvironment();
      const {cn: m, F: O} = s;
      const [C, M] = t("showNativeOverlaidScrollbars");
      const [R, V] = t("scrollbars.theme");
      const [L, U] = t("scrollbars.visibility");
      const [P, N] = t("scrollbars.autoHide");
      const [q, j] = t("scrollbars.autoHideSuspend");
      const [B] = t("scrollbars.autoHideDelay");
      const [F, X] = t("scrollbars.dragScroll");
      const [Y, W] = t("scrollbars.clickScroll");
      const [G, J] = t("overflow");
      const K = h && !n;
      const Q = O.x || O.y;
      const Z = r || d || v || g || n;
      const tt = p || U || J;
      const nt = C && y.x && y.y;
      const ot = !S && !k;
      const st = nt || ot;
      const setScrollbarVisibility = (t, n, o) => {
        const s = t.includes(z) && (L === H || L === "auto" && n === z);
        E(gt, s, o);
        return s;
      };
      _ = B;
      if (K) {
        if (q && Q) {
          manageAutoHideSuspension(false);
          f();
          b((() => {
            f = addEventListener($, "scroll", bind(manageAutoHideSuspension, true), {
              T: true
            });
          }));
        } else {
          manageAutoHideSuspension(true);
        }
      }
      if (M || ot) {
        E(at, st);
      }
      if (V) {
        E(u);
        E(R, true);
        u = R;
      }
      if (j && !q) {
        manageAutoHideSuspension(true);
      }
      if (N) {
        l = P === "move";
        i = P === "leave";
        a = P === "never";
        manageScrollbarsAutoHide(a, true);
      }
      if (X) {
        E(Ct, F);
      }
      if (W) {
        E(Ot, !!Y);
      }
      if (tt) {
        const t = setScrollbarVisibility(G.x, m.x, true);
        const n = setScrollbarVisibility(G.y, m.y, false);
        const o = t && n;
        E(ht, !o);
      }
      if (Z) {
        A();
        I();
        D();
        if (v) {
          T();
        }
        E(wt, !O.x, true);
        E(wt, !O.y, false);
        E(ft, w && !x);
      }
    }, {}, m ];
  };
  const createStructureSetupElements = t => {
    const n = getEnvironment();
    const {K: s, U: e} = n;
    const {elements: c} = s();
    const {padding: r, viewport: l, content: i} = c;
    const a = isHTMLElement(t);
    const u = a ? {} : t;
    const {elements: f} = u;
    const {padding: _, viewport: d, content: p} = f || {};
    const v = a ? t : u.target;
    const g = isBodyElement(v);
    const h = v.ownerDocument;
    const b = h.documentElement;
    const getDocumentWindow = () => h.defaultView || o;
    const w = bind(staticInitializationElement, [ v ]);
    const y = bind(dynamicInitializationElement, [ v ]);
    const S = bind(createDiv, "");
    const m = bind(w, S, l);
    const $ = bind(y, S, i);
    const elementHasOverflow = t => {
      const n = T(t);
      const o = M(t);
      const s = getStyles(t, O);
      const e = getStyles(t, C);
      return o.w - n.w > 0 && !overflowIsVisible(s) || o.h - n.h > 0 && !overflowIsVisible(e);
    };
    const x = m(d);
    const H = x === v;
    const E = H && g;
    const z = !H && $(p);
    const I = !H && x === z;
    const A = E ? b : x;
    const D = E ? A : v;
    const k = !H && y(S, r, _);
    const R = !I && z;
    const V = [ R, A, k, D ].map((t => isHTMLElement(t) && !parent(t) && t));
    const elementIsGenerated = t => t && inArray(V, t);
    const L = !elementIsGenerated(A) && elementHasOverflow(A) ? A : v;
    const U = E ? b : A;
    const P = E ? h : A;
    const q = {
      dt: v,
      vt: D,
      L: A,
      rn: k,
      ht: R,
      gt: U,
      Kt: P,
      ln: g ? b : L,
      Jt: h,
      bt: g,
      Tt: a,
      V: H,
      an: getDocumentWindow,
      wt: t => hasAttrClass(A, X, t),
      yt: (t, n) => addRemoveAttrClass(A, X, t, n),
      St: () => addRemoveAttrClass(U, X, K, true)
    };
    const {dt: Y, vt: W, rn: G, L: J, ht: Z} = q;
    const ot = [ () => {
      removeAttrs(W, [ B, N ]);
      removeAttrs(Y, N);
      if (g) {
        removeAttrs(b, [ N, B ]);
      }
    } ];
    let st = contents([ Z, J, G, W, Y ].find((t => t && !elementIsGenerated(t))));
    const et = E ? Y : Z || J;
    const ct = bind(runEachAndClear, ot);
    const appendElements = () => {
      const t = getDocumentWindow();
      const n = getFocusedElement();
      const unwrap = t => {
        appendChildren(parent(t), contents(t));
        removeElements(t);
      };
      const prepareWrapUnwrapFocus = t => addEventListener(t, "focusin focusout focus blur", stopAndPrevent, {
        A: true,
        I: false
      });
      const o = "tabindex";
      const s = getAttr(J, o);
      const c = prepareWrapUnwrapFocus(n);
      setAttrs(W, B, H ? "" : F);
      setAttrs(G, tt, "");
      setAttrs(J, X, "");
      setAttrs(Z, nt, "");
      if (!H) {
        setAttrs(J, o, s || "-1");
        if (g) {
          setAttrs(b, j, "");
        }
      }
      appendChildren(et, st);
      appendChildren(W, G);
      appendChildren(G || W, !H && J);
      appendChildren(J, Z);
      push(ot, [ c, () => {
        const t = getFocusedElement();
        const n = elementIsGenerated(J);
        const e = n && t === J ? Y : t;
        const c = prepareWrapUnwrapFocus(e);
        removeAttrs(G, tt);
        removeAttrs(Z, nt);
        removeAttrs(J, X);
        if (g) {
          removeAttrs(b, j);
        }
        if (s) {
          setAttrs(J, o, s);
        } else {
          removeAttrs(J, o);
        }
        if (elementIsGenerated(Z)) {
          unwrap(Z);
        }
        if (n) {
          unwrap(J);
        }
        if (elementIsGenerated(G)) {
          unwrap(G);
        }
        focusElement(e);
        c();
      } ]);
      if (e && !H) {
        addAttrClass(J, X, Q);
        push(ot, bind(removeAttrs, J, X));
      }
      focusElement(!H && g && n === Y && t.top === t ? J : n);
      c();
      st = 0;
      return ct;
    };
    return [ q, appendElements, ct ];
  };
  const createTrinsicUpdateSegment = ({ht: t}) => ({Qt: n, un: o, At: s}) => {
    const {$t: e} = n || {};
    const {Ot: c} = o;
    const r = t && (e || s);
    if (r) {
      setStyles(t, {
        [x]: c && "100%"
      });
    }
  };
  const createPaddingUpdateSegment = ({vt: t, rn: n, L: o, V: s}, e) => {
    const [c, r] = createCache({
      i: equalTRBL,
      o: topRightBottomLeft()
    }, bind(topRightBottomLeft, t, "padding", ""));
    return ({zt: t, Qt: l, un: i, At: a}) => {
      let [u, f] = r(a);
      const {U: _} = getEnvironment();
      const {_t: d, xt: p, Ct: v} = l || {};
      const {j: O} = i;
      const [C, x] = t("paddingAbsolute");
      const H = a || p;
      if (d || f || H) {
        [u, f] = c(a);
      }
      const E = !s && (x || v || f);
      if (E) {
        const t = !C || !n && !_;
        const s = u.r + u.l;
        const c = u.t + u.b;
        const r = {
          [S]: t && !O ? -s : 0,
          [m]: t ? -c : 0,
          [y]: t && O ? -s : 0,
          top: t ? -u.t : 0,
          right: t ? O ? -u.r : "auto" : 0,
          left: t ? O ? "auto" : -u.l : 0,
          [$]: t && `calc(100% + ${s}px)`
        };
        const l = {
          [g]: t ? u.t : 0,
          [h]: t ? u.r : 0,
          [w]: t ? u.b : 0,
          [b]: t ? u.l : 0
        };
        setStyles(n || o, r);
        setStyles(o, l);
        assignDeep(e, {
          rn: u,
          fn: !t,
          B: n ? l : assignDeep({}, r, l)
        });
      }
      return {
        _n: E
      };
    };
  };
  const createOverflowUpdateSegment = (t, n) => {
    const e = getEnvironment();
    const {vt: c, rn: l, L: i, V: u, Kt: f, gt: _, bt: d, yt: p, an: v} = t;
    const {U: g} = e;
    const h = d && u;
    const b = bind(s, 0);
    const w = {
      display: () => false,
      direction: t => t !== "ltr",
      flexDirection: t => t.endsWith("-reverse"),
      writingMode: t => t !== "horizontal-tb"
    };
    const y = keys(w);
    const S = {
      i: equalWH,
      o: {
        w: 0,
        h: 0
      }
    };
    const m = {
      i: equalXY,
      o: {}
    };
    const setMeasuringMode = t => {
      p(J, !h && t);
    };
    const getFlowDirectionStyles = () => getStyles(i, y);
    const getMeasuredScrollCoordinates = (t, n) => {
      const o = !keys(t).length;
      const s = !n && y.some((n => {
        const o = t[n];
        return isString(o) && w[n](o);
      }));
      const e = o && !s;
      if (e || !hasDimensions(i)) {
        return {
          D: {
            x: 0,
            y: 0
          },
          M: {
            x: 1,
            y: 1
          }
        };
      }
      setMeasuringMode(true);
      const c = getElementScroll(_);
      const l = addEventListener(f, z, (t => {
        const n = getElementScroll(_);
        if (t.isTrusted && n.x === c.x && n.y === c.y) {
          stopPropagation(t);
        }
      }), {
        A: true,
        T: true
      });
      const u = p(Z, true);
      scrollElementTo(_, {
        x: 0,
        y: 0
      });
      u();
      const d = getElementScroll(_);
      const v = M(_);
      scrollElementTo(_, {
        x: v.w,
        y: v.h
      });
      const g = getElementScroll(_);
      const h = {
        x: g.x - d.x,
        y: g.y - d.y
      };
      scrollElementTo(_, {
        x: -v.w,
        y: -v.h
      });
      const b = getElementScroll(_);
      const S = {
        x: b.x - d.x,
        y: b.y - d.y
      };
      const m = {
        x: r(h.x) >= r(S.x) ? g.x : b.x,
        y: r(h.y) >= r(S.y) ? g.y : b.y
      };
      scrollElementTo(_, c);
      a((() => l()));
      return {
        D: d,
        M: m
      };
    };
    const getOverflowAmount = (t, n) => {
      const s = o.devicePixelRatio % 1 !== 0 ? 1 : 0;
      const e = {
        w: b(t.w - n.w),
        h: b(t.h - n.h)
      };
      return {
        w: e.w > s ? e.w : 0,
        h: e.h > s ? e.h : 0
      };
    };
    const getViewportOverflowStyle = (t, n) => {
      const getAxisOverflowStyle = (t, n, o, s) => {
        const e = t === H ? E : overflowBehaviorToOverflowStyle(t);
        const c = overflowIsVisible(t);
        const r = overflowIsVisible(o);
        if (!n && !s) {
          return E;
        }
        if (c && r) {
          return H;
        }
        if (c) {
          const t = n ? H : E;
          return n && s ? e : t;
        }
        const l = r && s ? H : E;
        return n ? e : l;
      };
      return {
        x: getAxisOverflowStyle(n.x, t.x, n.y, t.y),
        y: getAxisOverflowStyle(n.y, t.y, n.x, t.x)
      };
    };
    const setViewportOverflowStyle = t => {
      const createAllOverflowStyleClassNames = t => [ H, E, z ].map((n => createViewportOverflowStyleClassName(overflowCssValueToOverflowStyle(n), t)));
      const n = createAllOverflowStyleClassNames(true).concat(createAllOverflowStyleClassNames()).join(" ");
      p(n);
      p(keys(t).map((n => createViewportOverflowStyleClassName(t[n], n === "x"))).join(" "), true);
    };
    const [O, C] = createCache(S, bind(getFractionalSize, i));
    const [$, x] = createCache(S, bind(M, i));
    const [I, A] = createCache(S);
    const [T] = createCache(m);
    const [k, R] = createCache(S);
    const [V] = createCache(m);
    const [L] = createCache({
      i: (t, n) => equal(t, n, deduplicateArray(concat(keys(t), keys(n)))),
      o: {}
    });
    const [U, P] = createCache({
      i: (t, n) => equalXY(t.D, n.D) && equalXY(t.M, n.M),
      o: getZeroScrollCoordinates()
    });
    const N = getStaticPluginModuleInstance(Ht);
    const createViewportOverflowStyleClassName = (t, n) => {
      const o = n ? Y : W;
      return `${o}${capitalizeFirstLetter(t)}`;
    };
    return ({zt: o, Qt: s, un: r, At: a}, {_n: u}) => {
      const {_t: f, Ht: _, xt: d, Ct: w, ft: y, Et: S} = s || {};
      const m = N && N.R(t, n, r, e, o);
      const {X: H, Y: E, W: z} = m || {};
      const [M, j] = getShowNativeOverlaidScrollbars(o, e);
      const [F, X] = o("overflow");
      const Y = overflowIsVisible(F.x);
      const W = overflowIsVisible(F.y);
      const G = f || u || d || w || S || j;
      let J = C(a);
      let K = x(a);
      let Z = A(a);
      let nt = R(a);
      if (j && g) {
        p(Q, !M);
      }
      if (G) {
        if (hasAttrClass(c, B, q)) {
          setMeasuringMode(true);
        }
        const t = E && E();
        const [n] = J = O(a);
        const [o] = K = $(a);
        const s = D(i);
        const e = h && getWindowSize(v());
        const r = {
          w: b(o.w + n.w),
          h: b(o.h + n.h)
        };
        const l = {
          w: b((e ? e.w : s.w + b(s.w - o.w)) + n.w),
          h: b((e ? e.h : s.h + b(s.h - o.h)) + n.h)
        };
        if (t) {
          t();
        }
        nt = k(l);
        Z = I(getOverflowAmount(r, l), a);
      }
      const [ot, st] = nt;
      const [et, ct] = Z;
      const [rt, lt] = K;
      const [it, at] = J;
      const [ut, ft] = T({
        x: et.w > 0,
        y: et.h > 0
      });
      const _t = Y && W && (ut.x || ut.y) || Y && ut.x && !ut.y || W && ut.y && !ut.x;
      const dt = u || w || S || at || lt || st || ct || X || j || G || _ && h;
      const [pt] = o("update.flowDirectionStyles");
      const [vt, gt] = L(pt ? pt(i) : getFlowDirectionStyles(), a);
      const ht = w || y || gt || ft || a;
      const [bt, wt] = ht ? U(getMeasuredScrollCoordinates(vt, !!pt), a) : P();
      let yt = getViewportOverflowStyle(ut, F);
      setMeasuringMode(false);
      if (dt) {
        setViewportOverflowStyle(yt);
        yt = getElementOverflowStyle(i, ut);
        if (z && H) {
          H(yt, rt, it);
          setStyles(i, z(yt));
        }
      }
      const [St, mt] = V(yt);
      addRemoveAttrClass(c, B, q, _t);
      addRemoveAttrClass(l, tt, q, _t);
      assignDeep(n, {
        cn: St,
        Vt: {
          x: ot.w,
          y: ot.h
        },
        Rt: {
          x: et.w,
          y: et.h
        },
        F: ut,
        Dt: sanitizeScrollCoordinates(bt, et)
      });
      return {
        sn: mt,
        tn: st,
        nn: ct,
        en: wt || ct
      };
    };
  };
  const createStructureSetup = t => {
    const [n, o, s] = createStructureSetupElements(t);
    const e = {
      rn: {
        t: 0,
        r: 0,
        b: 0,
        l: 0
      },
      fn: false,
      B: {
        [S]: 0,
        [m]: 0,
        [y]: 0,
        [g]: 0,
        [h]: 0,
        [w]: 0,
        [b]: 0
      },
      Vt: {
        x: 0,
        y: 0
      },
      Rt: {
        x: 0,
        y: 0
      },
      cn: {
        x: E,
        y: E
      },
      F: {
        x: false,
        y: false
      },
      Dt: getZeroScrollCoordinates()
    };
    const {dt: c, gt: r, V: l, St: i} = n;
    const {U: a, k: u} = getEnvironment();
    const f = !a && (u.x || u.y);
    const _ = [ createTrinsicUpdateSegment(n), createPaddingUpdateSegment(n, e), createOverflowUpdateSegment(n, e) ];
    return [ o, t => {
      const n = {};
      const o = f;
      const s = o && getElementScroll(r);
      const e = s && i();
      each(_, (o => {
        assignDeep(n, o(t, n) || {});
      }));
      scrollElementTo(r, s);
      if (e) {
        e();
      }
      if (!l) {
        scrollElementTo(c, 0);
      }
      return n;
    }, e, n, s ];
  };
  const createSetups = (t, n, o, s, e) => {
    let c = false;
    const r = createOptionCheck(n, {});
    const [l, i, a, u, f] = createStructureSetup(t);
    const [_, d, p] = createObserversSetup(u, a, r, (t => {
      update({}, t);
    }));
    const [v, g, , h] = createScrollbarsSetup(t, n, p, a, u, e);
    const updateHintsAreTruthy = t => keys(t).some((n => !!t[n]));
    const update = (t, e) => {
      if (o()) {
        return false;
      }
      const {dn: r, At: l, It: a, pn: u} = t;
      const f = r || {};
      const _ = !!l || !c;
      const v = {
        zt: createOptionCheck(n, f, _),
        dn: f,
        At: _
      };
      if (u) {
        g(v);
        return false;
      }
      const h = e || d(assignDeep({}, v, {
        It: a
      }));
      const b = i(assignDeep({}, v, {
        un: p,
        Qt: h
      }));
      g(assignDeep({}, v, {
        Qt: h,
        Zt: b
      }));
      const w = updateHintsAreTruthy(h);
      const y = updateHintsAreTruthy(b);
      const S = w || y || !isEmptyObject(f) || _;
      c = true;
      if (S) {
        s(t, {
          Qt: h,
          Zt: b
        });
      }
      return S;
    };
    return [ () => {
      const {ln: t, gt: n, St: o} = u;
      const s = getElementScroll(t);
      const e = [ _(), l(), v() ];
      const c = o();
      scrollElementTo(n, s);
      c();
      return bind(runEachAndClear, e);
    }, update, () => ({
      vn: p,
      gn: a
    }), {
      hn: u,
      bn: h
    }, f ];
  };
  const Lt = new WeakMap;
  const addInstance = (t, n) => {
    Lt.set(t, n);
  };
  const removeInstance = t => {
    Lt.delete(t);
  };
  const getInstance = t => Lt.get(t);
  const OverlayScrollbars = (t, n, o) => {
    const {tt: s} = getEnvironment();
    const e = isHTMLElement(t);
    const c = e ? t : t.target;
    const r = getInstance(c);
    if (n && !r) {
      let r = false;
      const l = [];
      const i = {};
      const validateOptions = t => {
        const n = removeUndefinedProperties(t);
        const o = getStaticPluginModuleInstance(V);
        return o ? o(n, true) : n;
      };
      const a = assignDeep({}, s(), validateOptions(n));
      const [u, f, _] = createEventListenerHub();
      const [d, p, v] = createEventListenerHub(o);
      const triggerEvent = (t, n) => {
        v(t, n);
        _(t, n);
      };
      const [g, h, b, w, y] = createSetups(t, a, (() => r), (({dn: t, At: n}, {Qt: o, Zt: s}) => {
        const {_t: e, Ct: c, $t: r, xt: l, Ht: i, ft: a} = o;
        const {tn: u, nn: f, sn: _, en: d} = s;
        triggerEvent("updated", [ S, {
          updateHints: {
            sizeChanged: !!e,
            directionChanged: !!c,
            heightIntrinsicChanged: !!r,
            overflowEdgeChanged: !!u,
            overflowAmountChanged: !!f,
            overflowStyleChanged: !!_,
            scrollCoordinatesChanged: !!d,
            contentMutation: !!l,
            hostMutation: !!i,
            appear: !!a
          },
          changedOptions: t || {},
          force: !!n
        } ]);
      }), (t => triggerEvent("scroll", [ S, t ])));
      const destroy = t => {
        removeInstance(c);
        runEachAndClear(l);
        r = true;
        triggerEvent("destroyed", [ S, t ]);
        f();
        p();
      };
      const S = {
        options(t, n) {
          if (t) {
            const o = n ? s() : {};
            const e = getOptionsDiff(a, assignDeep(o, validateOptions(t)));
            if (!isEmptyObject(e)) {
              assignDeep(a, e);
              h({
                dn: e
              });
            }
          }
          return assignDeep({}, a);
        },
        on: d,
        off: (t, n) => {
          if (t && n) {
            p(t, n);
          }
        },
        state() {
          const {vn: t, gn: n} = b();
          const {j: o} = t;
          const {Vt: s, Rt: e, cn: c, F: l, rn: i, fn: a, Dt: u} = n;
          return assignDeep({}, {
            overflowEdge: s,
            overflowAmount: e,
            overflowStyle: c,
            hasOverflow: l,
            scrollCoordinates: {
              start: u.D,
              end: u.M
            },
            padding: i,
            paddingAbsolute: a,
            directionRTL: o,
            destroyed: r
          });
        },
        elements() {
          const {dt: t, vt: n, rn: o, L: s, ht: e, gt: c, Kt: r} = w.hn;
          const {Ft: l, Gt: i} = w.bn;
          const translateScrollbarStructure = t => {
            const {Mt: n, Ut: o, Lt: s} = t;
            return {
              scrollbar: s,
              track: o,
              handle: n
            };
          };
          const translateScrollbarsSetupElement = t => {
            const {Xt: n, Yt: o} = t;
            const s = translateScrollbarStructure(n[0]);
            return assignDeep({}, s, {
              clone: () => {
                const t = translateScrollbarStructure(o());
                h({
                  pn: true
                });
                return t;
              }
            });
          };
          return assignDeep({}, {
            target: t,
            host: n,
            padding: o || s,
            viewport: s,
            content: e || s,
            scrollOffsetElement: c,
            scrollEventElement: r,
            scrollbarHorizontal: translateScrollbarsSetupElement(l),
            scrollbarVertical: translateScrollbarsSetupElement(i)
          });
        },
        update: t => h({
          At: t,
          It: true
        }),
        destroy: bind(destroy, false),
        plugin: t => i[keys(t)[0]]
      };
      push(l, [ y ]);
      addInstance(c, S);
      registerPluginModuleInstances(k, OverlayScrollbars, [ S, u, i ]);
      if (cancelInitialization(w.hn.bt, !e && t.cancel)) {
        destroy(true);
        return S;
      }
      push(l, g());
      triggerEvent("initialized", [ S ]);
      S.update();
      return S;
    }
    return r;
  };
  OverlayScrollbars.plugin = t => {
    const n = isArray(t);
    const o = n ? t : [ t ];
    const s = o.map((t => registerPluginModuleInstances(t, OverlayScrollbars)[0]));
    addPlugins(o);
    return n ? s : s[0];
  };
  OverlayScrollbars.valid = t => {
    const n = t && t.elements;
    const o = isFunction(n) && n();
    return isPlainObject(o) && !!getInstance(o.target);
  };
  OverlayScrollbars.env = () => {
    const {P: t, k: n, U: o, G: s, ot: e, st: c, K: r, Z: l, tt: i, nt: a} = getEnvironment();
    return assignDeep({}, {
      scrollbarsSize: t,
      scrollbarsOverlaid: n,
      scrollbarsHiding: o,
      scrollTimeline: s,
      staticDefaultInitialization: e,
      staticDefaultOptions: c,
      getDefaultInitialization: r,
      setDefaultInitialization: l,
      getDefaultOptions: i,
      setDefaultOptions: a
    });
  };
  OverlayScrollbars.nonce = setNonce;
  OverlayScrollbars.trustedTypePolicy = setTrustedTypePolicy;
  t.ClickScrollPlugin = It;
  t.OverlayScrollbars = OverlayScrollbars;
  t.ScrollbarsHidingPlugin = Et;
  t.SizeObserverPlugin = xt;
  Object.defineProperty(t, Symbol.toStringTag, {
    value: "Module"
  });
  return t;
}({});
//# sourceMappingURL=overlayscrollbars.browser.es6.js.map

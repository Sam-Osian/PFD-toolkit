/*!
 * OverlayScrollbars
 * Version: 2.14.0
 *
 * Copyright (c) Rene Haas | KingSora.
 * https://github.com/KingSora
 *
 * Released under the MIT license.
 */
var OverlayScrollbarsGlobal = function(r) {
  "use strict";
  var e = function createCache(r, e) {
    var a = r.i, n = r.v, t = r.o;
    var i = a;
    var v;
    var o = function cacheUpdateContextual(r, e) {
      var a = i;
      var o = r;
      var u = e || (n ? !n(a, o) : a !== o);
      if (u || t) {
        i = o;
        v = a;
      }
      return [ i, u, v ];
    };
    var u = function cacheUpdateIsolated(r) {
      return o(e(i, v), r);
    };
    var c = function getCurrentCache(r) {
      return [ i, !!r, v ];
    };
    return [ e ? u : o, c ];
  };
  var a = typeof window !== "undefined" && typeof HTMLElement !== "undefined" && !!window.document;
  var n = a ? window : {};
  var t = Math.max;
  var i = Math.min;
  var v = Math.round;
  var o = Math.abs;
  var u = Math.sign;
  var c = n.cancelAnimationFrame;
  var f = n.requestAnimationFrame;
  var l = n.setTimeout;
  var s = n.clearTimeout;
  var d = function getApi(r) {
    return typeof n[r] !== "undefined" ? n[r] : void 0;
  };
  var p = d("MutationObserver");
  var _ = d("IntersectionObserver");
  var g = d("ResizeObserver");
  var h = d("ScrollTimeline");
  var b = function isUndefined(r) {
    return r === void 0;
  };
  var m = function isNull(r) {
    return r === null;
  };
  var y = function type(r) {
    return b(r) || m(r) ? "" + r : Object.prototype.toString.call(r).replace(/^\[object (.+)\]$/, "$1").toLowerCase();
  };
  var S = function isNumber(r) {
    return typeof r === "number";
  };
  var w = function isString(r) {
    return typeof r === "string";
  };
  var O = function isBoolean(r) {
    return typeof r === "boolean";
  };
  var C = function isFunction(r) {
    return typeof r === "function";
  };
  var x = function isArray(r) {
    return Array.isArray(r);
  };
  var E = function isObject(r) {
    return typeof r === "object" && !x(r) && !m(r);
  };
  var A = function isArrayLike(r) {
    var e = !!r && r.length;
    var a = S(e) && e > -1 && e % 1 == 0;
    return x(r) || !C(r) && a ? e > 0 && E(r) ? e - 1 in r : true : false;
  };
  var T = function isPlainObject(r) {
    return !!r && r.constructor === Object;
  };
  var H = function isHTMLElement(r) {
    return r instanceof HTMLElement;
  };
  var P = function isElement(r) {
    return r instanceof Element;
  };
  var D = function animationCurrentTime() {
    return performance.now();
  };
  var z = function animateNumber(r, e, a, n, i) {
    var v = 0;
    var o = D();
    var u = t(0, a);
    var l = function frame(a) {
      var c = D();
      var s = c - o;
      var d = s >= u;
      var p = a ? 1 : 1 - (t(0, o + u - c) / u || 0);
      var _ = (e - r) * (C(i) ? i(p, p * u, 0, 1, u) : p) + r;
      var g = d || p === 1;
      if (n) {
        n(_, p, g);
      }
      v = g ? 0 : f((function() {
        return l();
      }));
    };
    l();
    return function(r) {
      c(v);
      if (r) {
        l(r);
      }
    };
  };
  function each(r, e) {
    if (A(r)) {
      for (var a = 0; a < r.length; a++) {
        if (e(r[a], a, r) === false) {
          break;
        }
      }
    } else if (r) {
      each(Object.keys(r), (function(a) {
        return e(r[a], a, r);
      }));
    }
    return r;
  }
  var M = function inArray(r, e) {
    return r.indexOf(e) >= 0;
  };
  var I = function concat(r, e) {
    return r.concat(e);
  };
  var L = function push(r, e, a) {
    if (!w(e) && A(e)) {
      Array.prototype.push.apply(r, e);
    } else {
      r.push(e);
    }
    return r;
  };
  var k = function from(r) {
    return Array.from(r || []);
  };
  var V = function createOrKeepArray(r) {
    if (x(r)) {
      return r;
    }
    return !w(r) && A(r) ? k(r) : [ r ];
  };
  var R = function isEmptyArray(r) {
    return !!r && !r.length;
  };
  var F = function deduplicateArray(r) {
    return k(new Set(r));
  };
  var N = function runEachAndClear(r, e, a) {
    var n = function runFn(r) {
      return r ? r.apply(void 0, e || []) : true;
    };
    each(r, n);
    if (!a) {
      r.length = 0;
    }
  };
  var j = "paddingTop";
  var U = "paddingRight";
  var q = "paddingLeft";
  var B = "paddingBottom";
  var Y = "marginLeft";
  var W = "marginRight";
  var X = "marginBottom";
  var Z = "overflowX";
  var G = "overflowY";
  var $ = "width";
  var J = "height";
  var K = "visible";
  var Q = "hidden";
  var rr = "scroll";
  var er = function capitalizeFirstLetter(r) {
    var e = String(r || "");
    return e ? e[0].toUpperCase() + e.slice(1) : "";
  };
  var ar = function equal(r, e, a, n) {
    if (r && e) {
      var t = true;
      each(a, (function(a) {
        var n = r[a];
        var i = e[a];
        if (n !== i) {
          t = false;
        }
      }));
      return t;
    }
    return false;
  };
  var nr = function equalWH(r, e) {
    return ar(r, e, [ "w", "h" ]);
  };
  var tr = function equalXY(r, e) {
    return ar(r, e, [ "x", "y" ]);
  };
  var ir = function equalTRBL(r, e) {
    return ar(r, e, [ "t", "r", "b", "l" ]);
  };
  var vr = function bind(r) {
    for (var e = arguments.length, a = new Array(e > 1 ? e - 1 : 0), n = 1; n < e; n++) {
      a[n - 1] = arguments[n];
    }
    return r.bind.apply(r, [ 0 ].concat(a));
  };
  var or = function selfClearTimeout(r) {
    var e;
    var a = r ? l : f;
    var n = r ? s : c;
    return [ function(t) {
      n(e);
      e = a((function() {
        return t();
      }), C(r) ? r() : r);
    }, function() {
      return n(e);
    } ];
  };
  var ur = function getDebouncer(r) {
    var e = C(r) ? r() : r;
    if (S(e)) {
      var a = e ? l : f;
      var n = e ? s : c;
      return function(r) {
        var t = a((function() {
          return r();
        }), e);
        return function() {
          n(t);
        };
      };
    }
    return e && e.u;
  };
  var cr = function debounce(r, e) {
    var a = e || {}, n = a.p, t = a._, i = a.m, v = a.S;
    var o;
    var u;
    var c;
    var f;
    var l = function invokeFunctionToDebounce(e) {
      if (u) {
        u();
      }
      if (o) {
        o();
      }
      f = u = o = c = void 0;
      r.apply(this, e);
    };
    var s = function mergeParms(r) {
      return v && c ? v(c, r) : r;
    };
    var d = function flush() {
      if (u && c) {
        l(s(c) || c);
      }
    };
    var p = function debouncedFn() {
      var r = k(arguments);
      var e = ur(n);
      if (e) {
        var a = typeof i === "function" ? i() : i;
        var v = ur(t);
        var p = s(r);
        var _ = p || r;
        var g = l.bind(0, _);
        if (u) {
          u();
        }
        if (a && !f) {
          g();
          f = true;
          u = e((function() {
            return f = void 0;
          }));
        } else {
          u = e(g);
          if (v && !o) {
            o = v(d);
          }
        }
        c = _;
      } else {
        l(r);
      }
    };
    p.O = d;
    return p;
  };
  var fr = function hasOwnProperty(r, e) {
    return Object.prototype.hasOwnProperty.call(r, e);
  };
  var lr = function keys(r) {
    return r ? Object.keys(r) : [];
  };
  var sr = function assignDeep(r, e, a, n, t, i, v) {
    var o = [ e, a, n, t, i, v ];
    if ((typeof r !== "object" || m(r)) && !C(r)) {
      r = {};
    }
    each(o, (function(e) {
      each(e, (function(a, n) {
        var t = e[n];
        if (r === t) {
          return true;
        }
        var i = x(t);
        if (t && T(t)) {
          var v = r[n];
          var o = v;
          if (i && !x(v)) {
            o = [];
          } else if (!i && !T(v)) {
            o = {};
          }
          r[n] = sr(o, t);
        } else {
          r[n] = i ? t.slice() : t;
        }
      }));
    }));
    return r;
  };
  var dr = function removeUndefinedProperties(r, e) {
    return each(sr({}, r), (function(r, e, a) {
      if (r === void 0) {
        delete a[e];
      } else if (r && T(r)) {
        a[e] = dr(r);
      }
    }));
  };
  var pr = function isEmptyObject(r) {
    return !lr(r).length;
  };
  var _r = function noop() {};
  var gr = function capNumber(r, e, a) {
    return t(r, i(e, a));
  };
  var hr = function getDomTokensArray(r) {
    return F((x(r) ? r : (r || "").split(" ")).filter((function(r) {
      return r;
    })));
  };
  var br = function getAttr(r, e) {
    return r && r.getAttribute(e);
  };
  var mr = function hasAttr(r, e) {
    return r && r.hasAttribute(e);
  };
  var yr = function setAttrs(r, e, a) {
    each(hr(e), (function(e) {
      if (r) {
        r.setAttribute(e, String(a || ""));
      }
    }));
  };
  var Sr = function removeAttrs(r, e) {
    each(hr(e), (function(e) {
      return r && r.removeAttribute(e);
    }));
  };
  var wr = function domTokenListAttr(r, e) {
    var a = hr(br(r, e));
    var n = vr(yr, r, e);
    var t = function domTokenListOperation(r, e) {
      var n = new Set(a);
      each(hr(r), (function(r) {
        n[e](r);
      }));
      return k(n).join(" ");
    };
    return {
      C: function _remove(r) {
        return n(t(r, "delete"));
      },
      A: function _add(r) {
        return n(t(r, "add"));
      },
      T: function _has(r) {
        var e = hr(r);
        return e.reduce((function(r, e) {
          return r && a.includes(e);
        }), e.length > 0);
      }
    };
  };
  var Or = function removeAttrClass(r, e, a) {
    wr(r, e).C(a);
    return vr(Cr, r, e, a);
  };
  var Cr = function addAttrClass(r, e, a) {
    wr(r, e).A(a);
    return vr(Or, r, e, a);
  };
  var xr = function addRemoveAttrClass(r, e, a, n) {
    return (n ? Cr : Or)(r, e, a);
  };
  var Er = function hasAttrClass(r, e, a) {
    return wr(r, e).T(a);
  };
  var Ar = function createDomTokenListClass(r) {
    return wr(r, "class");
  };
  var Tr = function removeClass(r, e) {
    Ar(r).C(e);
  };
  var Hr = function addClass(r, e) {
    Ar(r).A(e);
    return vr(Tr, r, e);
  };
  var Pr = function find(r, e) {
    var a = e ? P(e) && e : document;
    return a ? k(a.querySelectorAll(r)) : [];
  };
  var Dr = function findFirst(r, e) {
    var a = e ? P(e) && e : document;
    return a && a.querySelector(r);
  };
  var zr = function is(r, e) {
    return P(r) && r.matches(e);
  };
  var Mr = function isBodyElement(r) {
    return zr(r, "body");
  };
  var Ir = function contents(r) {
    return r ? k(r.childNodes) : [];
  };
  var Lr = function parent(r) {
    return r && r.parentElement;
  };
  var kr = function closest(r, e) {
    return P(r) && r.closest(e);
  };
  var Vr = function getFocusedElement(r) {
    return document.activeElement;
  };
  var Rr = function liesBetween(r, e, a) {
    var n = kr(r, e);
    var t = r && Dr(a, n);
    var i = kr(t, e) === n;
    return n && t ? n === r || t === r || i && kr(kr(r, a), e) !== n : false;
  };
  var Fr = function removeElements(r) {
    each(V(r), (function(r) {
      var e = Lr(r);
      if (r && e) {
        e.removeChild(r);
      }
    }));
  };
  var Nr = function appendChildren(r, e) {
    return vr(Fr, r && e && each(V(e), (function(e) {
      if (e) {
        r.appendChild(e);
      }
    })));
  };
  var jr;
  var Ur = function getTrustedTypePolicy() {
    return jr;
  };
  var qr = function setTrustedTypePolicy(r) {
    jr = r;
  };
  var Br = function createDiv(r) {
    var e = document.createElement("div");
    yr(e, "class", r);
    return e;
  };
  var Yr = function createDOM(r) {
    var e = Br();
    var a = Ur();
    var n = r.trim();
    e.innerHTML = a ? a.createHTML(n) : n;
    return each(Ir(e), (function(r) {
      return Fr(r);
    }));
  };
  var Wr = function getCSSVal(r, e) {
    return r.getPropertyValue(e) || r[e] || "";
  };
  var Xr = function validFiniteNumber(r) {
    var e = r || 0;
    return isFinite(e) ? e : 0;
  };
  var Zr = function parseToZeroOrNumber(r) {
    return Xr(parseFloat(r || ""));
  };
  var Gr = function roundCssNumber(r) {
    return Math.round(r * 1e4) / 1e4;
  };
  var $r = function numberToCssPx(r) {
    return Gr(Xr(r)) + "px";
  };
  function setStyles(r, e) {
    r && e && each(e, (function(e, a) {
      try {
        var n = r.style;
        var t = m(e) || O(e) ? "" : S(e) ? $r(e) : e;
        if (a.indexOf("--") === 0) {
          n.setProperty(a, t);
        } else {
          n[a] = t;
        }
      } catch (i) {}
    }));
  }
  function getStyles(r, e, a) {
    var t = w(e);
    var i = t ? "" : {};
    if (r) {
      var v = n.getComputedStyle(r, a) || r.style;
      i = t ? Wr(v, e) : k(e).reduce((function(r, e) {
        r[e] = Wr(v, e);
        return r;
      }), i);
    }
    return i;
  }
  var Jr = function topRightBottomLeft(r, e, a) {
    var n = e ? e + "-" : "";
    var t = a ? "-" + a : "";
    var i = n + "top" + t;
    var v = n + "right" + t;
    var o = n + "bottom" + t;
    var u = n + "left" + t;
    var c = getStyles(r, [ i, v, o, u ]);
    return {
      t: Zr(c[i]),
      r: Zr(c[v]),
      b: Zr(c[o]),
      l: Zr(c[u])
    };
  };
  var Kr = function getTrasformTranslateValue(r, e) {
    return "translate" + (E(r) ? "(" + r.x + "," + r.y + ")" : (e ? "X" : "Y") + "(" + r + ")");
  };
  var Qr = function elementHasDimensions(r) {
    return !!(r.offsetWidth || r.offsetHeight || r.getClientRects().length);
  };
  var re = {
    w: 0,
    h: 0
  };
  var ee = function getElmWidthHeightProperty(r, e) {
    return e ? {
      w: e[r + "Width"],
      h: e[r + "Height"]
    } : re;
  };
  var ae = function getWindowSize(r) {
    return ee("inner", r || n);
  };
  var ne = vr(ee, "offset");
  var te = vr(ee, "client");
  var ie = vr(ee, "scroll");
  var ve = function getFractionalSize(r) {
    var e = parseFloat(getStyles(r, $)) || 0;
    var a = parseFloat(getStyles(r, J)) || 0;
    return {
      w: e - v(e),
      h: a - v(a)
    };
  };
  var oe = function getBoundingClientRect(r) {
    return r.getBoundingClientRect();
  };
  var ue = function hasDimensions(r) {
    return !!r && Qr(r);
  };
  var ce = function domRectHasDimensions(r) {
    return !!(r && (r[J] || r[$]));
  };
  var fe = function domRectAppeared(r, e) {
    var a = ce(r);
    var n = ce(e);
    return !n && a;
  };
  var le = function removeEventListener(r, e, a, n) {
    each(hr(e), (function(e) {
      if (r) {
        r.removeEventListener(e, a, n);
      }
    }));
  };
  var se = function addEventListener(r, e, a, n) {
    var t;
    var i = (t = n && n.H) != null ? t : true;
    var v = n && n.P || false;
    var o = n && n.D || false;
    var u = {
      passive: i,
      capture: v
    };
    return vr(N, hr(e).map((function(e) {
      var n = o ? function(t) {
        le(r, e, n, v);
        if (a) {
          a(t);
        }
      } : a;
      if (r) {
        r.addEventListener(e, n, u);
      }
      return vr(le, r, e, n, v);
    })));
  };
  var de = function stopPropagation(r) {
    return r.stopPropagation();
  };
  var pe = function preventDefault(r) {
    return r.preventDefault();
  };
  var _e = function stopAndPrevent(r) {
    return de(r) || pe(r);
  };
  var ge = function scrollElementTo(r, e) {
    var a = S(e) ? {
      x: e,
      y: e
    } : e || {}, n = a.x, t = a.y;
    S(n) && (r.scrollLeft = n);
    S(t) && (r.scrollTop = t);
  };
  var he = function getElementScroll(r) {
    return {
      x: r.scrollLeft,
      y: r.scrollTop
    };
  };
  var be = function getZeroScrollCoordinates() {
    return {
      M: {
        x: 0,
        y: 0
      },
      I: {
        x: 0,
        y: 0
      }
    };
  };
  var me = function sanitizeScrollCoordinates(r, e) {
    var a = r.M, n = r.I;
    var t = e.w, i = e.h;
    var v = function sanitizeAxis(r, e, a) {
      var n = u(r) * a;
      var t = u(e) * a;
      if (n === t) {
        var i = o(r);
        var v = o(e);
        t = i > v ? 0 : t;
        n = i < v ? 0 : n;
      }
      n = n === t ? 0 : n;
      return [ n + 0, t + 0 ];
    };
    var c = v(a.x, n.x, t), f = c[0], l = c[1];
    var s = v(a.y, n.y, i), d = s[0], p = s[1];
    return {
      M: {
        x: f,
        y: d
      },
      I: {
        x: l,
        y: p
      }
    };
  };
  var ye = function isDefaultDirectionScrollCoordinates(r) {
    var e = r.M, a = r.I;
    var n = function getAxis(r, e) {
      return r === 0 && r <= e;
    };
    return {
      x: n(e.x, a.x),
      y: n(e.y, a.y)
    };
  };
  var Se = function getScrollCoordinatesPercent(r, e) {
    var a = r.M, n = r.I;
    var t = function getAxis(r, e, a) {
      return gr(0, 1, (r - a) / (r - e) || 0);
    };
    return {
      x: t(a.x, n.x, e.x),
      y: t(a.y, n.y, e.y)
    };
  };
  var we = function focusElement(r) {
    if (r && r.focus) {
      r.focus({
        preventScroll: true,
        focusVisible: false
      });
    }
  };
  var Oe = function manageListener(r, e) {
    each(V(e), r);
  };
  var Ce = function createEventListenerHub(r) {
    var e = new Map;
    var a = function removeEvent(r, a) {
      if (r) {
        var n = e.get(r);
        Oe((function(r) {
          if (n) {
            n[r ? "delete" : "clear"](r);
          }
        }), a);
      } else {
        e.forEach((function(r) {
          r.clear();
        }));
        e.clear();
      }
    };
    var n = function addEvent(r, t) {
      if (w(r)) {
        var i = e.get(r) || new Set;
        e.set(r, i);
        Oe((function(r) {
          if (C(r)) {
            i.add(r);
          }
        }), t);
        return vr(a, r, t);
      }
      if (O(t) && t) {
        a();
      }
      var v = lr(r);
      var o = [];
      each(v, (function(e) {
        var a = r[e];
        if (a) {
          L(o, n(e, a));
        }
      }));
      return vr(N, o);
    };
    var t = function triggerEvent(r, a) {
      each(k(e.get(r)), (function(r) {
        if (a && !R(a)) {
          r.apply(0, a);
        } else {
          r();
        }
      }));
    };
    n(r || {});
    return [ n, a, t ];
  };
  var xe = {};
  var Ee = {};
  var Ae = function addPlugins(r) {
    each(r, (function(r) {
      return each(r, (function(e, a) {
        xe[a] = r[a];
      }));
    }));
  };
  var Te = function registerPluginModuleInstances(r, e, a) {
    return lr(r).map((function(n) {
      var t = r[n], i = t.static, v = t.instance;
      var o = a || [], u = o[0], c = o[1], f = o[2];
      var l = a ? v : i;
      if (l) {
        var s = a ? l(u, c, e) : l(e);
        return (f || Ee)[n] = s;
      }
    }));
  };
  var He = function getStaticPluginModuleInstance(r) {
    return Ee[r];
  };
  function getDefaultExportFromCjs(r) {
    return r && r.L && Object.prototype.hasOwnProperty.call(r, "default") ? r["default"] : r;
  }
  var Pe = {
    exports: {}
  };
  (function(r) {
    function _extends() {
      return r.exports = _extends = Object.assign ? Object.assign.bind() : function(r) {
        for (var e = 1; e < arguments.length; e++) {
          var a = arguments[e];
          for (var n in a) {
            ({}).hasOwnProperty.call(a, n) && (r[n] = a[n]);
          }
        }
        return r;
      }, r.exports.L = true, r.exports["default"] = r.exports, _extends.apply(null, arguments);
    }
    r.exports = _extends, r.exports.L = true, r.exports["default"] = r.exports;
  })(Pe);
  var De = Pe.exports;
  var ze = /*@__PURE__*/ getDefaultExportFromCjs(De);
  var Me = {
    boolean: "__TPL_boolean_TYPE__",
    number: "__TPL_number_TYPE__",
    string: "__TPL_string_TYPE__",
    array: "__TPL_array_TYPE__",
    object: "__TPL_object_TYPE__",
    function: "__TPL_function_TYPE__",
    null: "__TPL_null_TYPE__"
  };
  var Ie = function validateRecursive(r, e, a, n) {
    var t = {};
    var i = ze({}, e);
    var v = lr(r).filter((function(r) {
      return fr(e, r);
    }));
    each(v, (function(v) {
      var o = e[v];
      var u = r[v];
      var c = T(u);
      var f = n ? n + "." : "";
      if (c && T(o)) {
        var l = Ie(u, o, a, f + v), s = l[0], d = l[1];
        t[v] = s;
        i[v] = d;
        each([ i, t ], (function(r) {
          if (pr(r[v])) {
            delete r[v];
          }
        }));
      } else if (!c) {
        var p = false;
        var _ = [];
        var g = [];
        var h = y(o);
        var m = V(u);
        each(m, (function(r) {
          var e;
          each(Me, (function(a, n) {
            if (a === r) {
              e = n;
            }
          }));
          var a = b(e);
          if (a && w(o)) {
            var n = r.split(" ");
            p = !!n.find((function(r) {
              return r === o;
            }));
            L(_, n);
          } else {
            p = Me[h] === r;
          }
          L(g, a ? Me.string : e);
          return !p;
        }));
        if (p) {
          t[v] = o;
        } else if (a) {
          console.warn('The option "' + f + v + "\" wasn't set, because it doesn't accept the type [ " + h.toUpperCase() + ' ] with the value of "' + o + '".\r\n' + "Accepted types are: [ " + g.join(", ").toUpperCase() + " ].\r\n" + (_.length > 0 ? "\r\nValid strings are: [ " + _.join(", ") + " ]." : ""));
        }
        delete i[v];
      }
    }));
    return [ t, i ];
  };
  var Le = function validateOptions(r, e, a) {
    return Ie(r, e, a);
  };
  var ke = "__osOptionsValidationPlugin";
  /* @__PURE__ */  (function(r) {
    return r = {}, r[ke] = {
      static: function _static() {
        var r = Me.number;
        var e = Me.boolean;
        var a = [ Me.array, Me.null ];
        var n = "hidden scroll visible visible-hidden";
        var t = "visible hidden auto";
        var i = "never scroll leavemove";
        var v = [ e, Me.string ];
        var o = {
          paddingAbsolute: e,
          showNativeOverlaidScrollbars: e,
          update: {
            elementEvents: a,
            attributes: a,
            debounce: [ Me.number, Me.array, Me.object, Me.null ],
            ignoreMutation: [ Me.function, Me.null ],
            flowDirectionStyles: [ Me.function, Me.null ]
          },
          overflow: {
            x: n,
            y: n
          },
          scrollbars: {
            theme: [ Me.string, Me.null ],
            visibility: t,
            autoHide: i,
            autoHideDelay: r,
            autoHideSuspend: e,
            dragScroll: e,
            clickScroll: v,
            pointers: [ Me.array, Me.null ]
          }
        };
        return function(r, e) {
          var a = Le(o, r, e), n = a[0], t = a[1];
          return ze({}, t, n);
        };
      }
    }, r;
  })();
  var Ve = "data-overlayscrollbars";
  var Re = "os-environment";
  var Fe = Re + "-scrollbar-hidden";
  var Ne = Ve + "-initialize";
  var je = "noClipping";
  var Ue = Ve + "-body";
  var qe = Ve;
  var Be = "host";
  var Ye = Ve + "-viewport";
  var We = Z;
  var Xe = G;
  var Ze = "arrange";
  var Ge = "measuring";
  var $e = "scrolling";
  var Je = "scrollbarHidden";
  var Ke = "noContent";
  var Qe = Ve + "-padding";
  var ra = Ve + "-content";
  var ea = "os-size-observer";
  var aa = ea + "-appear";
  var na = ea + "-listener";
  var ta = na + "-scroll";
  var ia = na + "-item";
  var va = ia + "-final";
  var oa = "os-trinsic-observer";
  var ua = "os-theme-none";
  var ca = "os-scrollbar";
  var fa = ca + "-rtl";
  var la = ca + "-horizontal";
  var sa = ca + "-vertical";
  var da = ca + "-track";
  var pa = ca + "-handle";
  var _a = ca + "-visible";
  var ga = ca + "-cornerless";
  var ha = ca + "-interaction";
  var ba = ca + "-unusable";
  var ma = ca + "-auto-hide";
  var ya = ma + "-hidden";
  var Sa = ca + "-wheel";
  var wa = da + "-interactive";
  var Oa = pa + "-interactive";
  var Ca = "__osSizeObserverPlugin";
  var xa = /* @__PURE__ */ function(r) {
    return r = {}, r[Ca] = {
      static: function _static() {
        return function(r, e, a) {
          var n;
          var t = 3333333;
          var i = "scroll";
          var v = Yr('<div class="' + ia + '" dir="ltr"><div class="' + ia + '"><div class="' + va + '"></div></div><div class="' + ia + '"><div class="' + va + '" style="width: 200%; height: 200%"></div></div></div>');
          var o = v[0];
          var u = o.lastChild;
          var l = o.firstChild;
          var s = l == null ? void 0 : l.firstChild;
          var d = ne(o);
          var p = d;
          var _ = false;
          var g;
          var h = function reset() {
            ge(l, t);
            ge(u, t);
          };
          var b = function onResized(r) {
            g = 0;
            if (_) {
              d = p;
              e(r === true);
            }
          };
          var m = function onScroll(r) {
            p = ne(o);
            _ = !r || !nr(p, d);
            if (r) {
              de(r);
              if (_ && !g) {
                c(g);
                g = f(b);
              }
            } else {
              b(r === false);
            }
            h();
          };
          var y = [ Nr(r, v), se(l, i, m), se(u, i, m) ];
          Hr(r, ta);
          setStyles(s, (n = {}, n[$] = t, n[J] = t, n));
          f(h);
          return [ a ? vr(m, false) : h, y ];
        };
      }
    }, r;
  }();
  var Ea = function getShowNativeOverlaidScrollbars(r, e) {
    var a = e.k;
    var n = r("showNativeOverlaidScrollbars"), t = n[0], i = n[1];
    return [ t && a.x && a.y, i ];
  };
  var Aa = function overflowIsVisible(r) {
    return r.indexOf(K) === 0;
  };
  var Ta = function overflowBehaviorToOverflowStyle(r) {
    return r.replace(K + "-", "");
  };
  var Ha = function overflowCssValueToOverflowStyle(r, e) {
    if (r === "auto") {
      return e ? rr : Q;
    }
    var a = r || Q;
    return [ Q, rr, K ].includes(a) ? a : Q;
  };
  var Pa = function getElementOverflowStyle(r, e) {
    var a = getStyles(r, [ Z, G ]), n = a.overflowX, t = a.overflowY;
    return {
      x: Ha(n, e.x),
      y: Ha(t, e.y)
    };
  };
  var Da = "__osScrollbarsHidingPlugin";
  var za = /* @__PURE__ */ function(r) {
    return r = {}, r[Da] = {
      static: function _static() {
        return {
          V: function _viewportArrangement(r, e, a, n, t) {
            var i = r.R, v = r.F;
            var o = n.N, u = n.k, c = n.j;
            var f = !i && !o && (u.x || u.y);
            var l = Ea(t, n), s = l[0];
            var d = function _getViewportOverflowHideOffset(r) {
              var e = o || s ? 0 : 42;
              var a = function getHideOffsetPerAxis(r, a, n) {
                var t = r ? e : n;
                var i = a && !o ? t : 0;
                var v = r && !!e;
                return [ i, v ];
              };
              var n = a(u.x, r.x === rr, c.x), t = n[0], i = n[1];
              var v = a(u.y, r.y === rr, c.y), f = v[0], l = v[1];
              return {
                U: {
                  x: t,
                  y: f
                },
                q: {
                  x: i,
                  y: l
                }
              };
            };
            var p = function _hideNativeScrollbars(r) {
              if (!i) {
                var n;
                var t = a.B;
                var v = sr({}, (n = {}, n[W] = 0, n[X] = 0, n[Y] = 0, n));
                var o = d(r), u = o.U, c = o.q;
                var l = c.x, s = c.y;
                var p = u.x, _ = u.y;
                var g = e.Y;
                var h = t ? Y : W;
                var b = t ? q : U;
                var m = g[h];
                var y = g[X];
                var S = g[b];
                var w = g[B];
                v[$] = "calc(100% + " + (_ + m * -1) + "px)";
                v[h] = -_ + m;
                v[X] = -p + y;
                if (f) {
                  v[b] = S + (s ? _ : 0);
                  v[B] = w + (l ? p : 0);
                }
                return v;
              }
            };
            var _ = function _arrangeViewport(r, n, t) {
              if (f) {
                var i = e.Y;
                var o = d(r), u = o.U, c = o.q;
                var l = c.x, s = c.y;
                var p = u.x, _ = u.y;
                var g = a.B;
                var h = g ? U : q;
                var b = i[h];
                var m = i.paddingTop;
                var y = n.w + t.w;
                var S = n.h + t.h;
                var w = {
                  w: _ && s ? _ + y - b + "px" : "",
                  h: p && l ? p + S - m + "px" : ""
                };
                setStyles(v, {
                  "--os-vaw": w.w,
                  "--os-vah": w.h
                });
              }
              return f;
            };
            var g = function _undoViewportArrange() {
              if (f) {
                var r = e.W, a = e.Y;
                var n = Pa(v, r);
                var t = d(n), i = t.q;
                var o = i.x, u = i.y;
                var c = {};
                var l = function assignProps(r) {
                  return each(r, (function(r) {
                    c[r] = a[r];
                  }));
                };
                if (o) {
                  l([ X, j, B ]);
                }
                if (u) {
                  l([ Y, W, q, U ]);
                }
                var s = getStyles(v, lr(c));
                var _ = Or(v, Ye, Ze);
                setStyles(v, c);
                return function() {
                  setStyles(v, sr({}, s, p(n)));
                  _();
                };
              }
              return _r;
            };
            return {
              X: _,
              Z: g,
              G: p
            };
          }
        };
      }
    }, r;
  }();
  var Ma = "__osClickScrollPlugin";
  var Ia = /* @__PURE__ */ function(r) {
    return r = {}, r[Ma] = {
      static: function _static() {
        return function(r, e, a, n) {
          var t = false;
          var i = _r;
          var v = 133;
          var o = 222;
          var u = or(v), c = u[0], f = u[1];
          var l = Math.sign(e);
          var s = a * l;
          var d = s / 2;
          var p = function easing(r) {
            return 1 - (1 - r) * (1 - r);
          };
          var _ = function easedEndPressAnimation(e, a) {
            return z(e, a, o, r, p);
          };
          var g = function linearPressAnimation(a, n) {
            return z(a, e - s, v * n, (function(a, n, t) {
              r(a);
              if (t) {
                i = _(a, e);
              }
            }));
          };
          var h = z(0, s, o, (function(v, o, u) {
            r(v);
            if (u) {
              n(t);
              if (!t) {
                var f = e - v;
                var p = Math.sign(f - d) === l;
                if (p) {
                  c((function() {
                    var r = f - s;
                    var n = Math.sign(r) === l;
                    i = n ? g(v, Math.abs(r) / a) : _(v, e);
                  }));
                }
              }
            }
          }), p);
          return function(r) {
            t = true;
            if (r) {
              h();
            }
            f();
            i();
          };
        };
      }
    }, r;
  }();
  var La = function opsStringify(r) {
    return JSON.stringify(r, (function(r, e) {
      if (C(e)) {
        throw 0;
      }
      return e;
    }));
  };
  var ka = function getPropByPath(r, e) {
    return r ? ("" + e).split(".").reduce((function(r, e) {
      return r && fr(r, e) ? r[e] : void 0;
    }), r) : void 0;
  };
  var Va = [ 0, 33 ];
  var Ra = [ 33, 99 ];
  var Fa = [ 222, 666, true ];
  var Na = {
    paddingAbsolute: false,
    showNativeOverlaidScrollbars: false,
    update: {
      elementEvents: [ [ "img", "load" ] ],
      debounce: {
        mutation: Va,
        resize: null,
        event: Ra,
        env: Fa
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
  var ja = function getOptionsDiff(r, e) {
    var a = {};
    var n = I(lr(e), lr(r));
    each(n, (function(n) {
      var t = r[n];
      var i = e[n];
      if (E(t) && E(i)) {
        sr(a[n] = {}, ja(t, i));
        if (pr(a[n])) {
          delete a[n];
        }
      } else if (fr(e, n) && i !== t) {
        var v = true;
        if (x(t) || x(i)) {
          try {
            if (La(t) === La(i)) {
              v = false;
            }
          } catch (o) {}
        }
        if (v) {
          a[n] = i;
        }
      }
    }));
    return a;
  };
  var Ua = function createOptionCheck(r, e, a) {
    return function(n) {
      return [ ka(r, n), a || ka(e, n) !== void 0 ];
    };
  };
  var qa;
  var Ba = function getNonce() {
    return qa;
  };
  var Ya = function setNonce(r) {
    qa = r;
  };
  var Wa;
  var Xa = function createEnvironment() {
    var r = function getNativeScrollbarSize(r, e, a) {
      Nr(document.body, r);
      Nr(document.body, r);
      var n = te(r);
      var t = ne(r);
      var i = ve(e);
      if (a) {
        Fr(r);
      }
      return {
        x: t.h - n.h + i.h,
        y: t.w - n.w + i.w
      };
    };
    var a = function getNativeScrollbarsHiding(r) {
      var e = false;
      var a = Hr(r, Fe);
      try {
        e = getStyles(r, "scrollbar-width") === "none" || getStyles(r, "display", "::-webkit-scrollbar") === "none";
      } catch (n) {}
      a();
      return e;
    };
    var t = "." + Re + "{scroll-behavior:auto!important;position:fixed;opacity:0;visibility:hidden;overflow:scroll;height:200px;width:200px;z-index:-1}." + Re + " div{width:200%;height:200%;margin:10px 0}." + Fe + "{scrollbar-width:none!important}." + Fe + "::-webkit-scrollbar,." + Fe + "::-webkit-scrollbar-corner{appearance:none!important;display:none!important;width:0!important;height:0!important}";
    var i = Yr('<div class="' + Re + '"><div></div><style>' + t + "</style></div>");
    var v = i[0];
    var o = v.firstChild;
    var u = v.lastChild;
    var c = Ba();
    if (c) {
      u.nonce = c;
    }
    var f = Ce(), l = f[0], s = f[2];
    var d = e({
      i: r(v, o),
      v: tr
    }, vr(r, v, o, true)), p = d[0], _ = d[1];
    var g = _(), b = g[0];
    var m = a(v);
    var y = {
      x: b.x === 0,
      y: b.y === 0
    };
    var S = {
      elements: {
        host: null,
        padding: !m,
        viewport: function viewport(r) {
          return m && Mr(r) && r;
        },
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
    var w = sr({}, Na);
    var O = vr(sr, {}, w);
    var x = vr(sr, {}, S);
    var E = {
      j: b,
      k: y,
      N: m,
      $: !!h,
      J: vr(l, "r"),
      K: x,
      rr: function _setDefaultInitialization(r) {
        return sr(S, r) && x();
      },
      er: O,
      ar: function _setDefaultOptions(r) {
        return sr(w, r) && O();
      },
      nr: sr({}, S),
      tr: sr({}, w)
    };
    Sr(v, "style");
    Fr(v);
    se(n, "resize", (function() {
      s("r", []);
    }));
    if (C(n.matchMedia) && !m && (!y.x || !y.y)) {
      var A = function addZoomListener(r) {
        var e = n.matchMedia("(resolution: " + n.devicePixelRatio + "dppx)");
        se(e, "change", (function() {
          r();
          A(r);
        }), {
          D: true
        });
      };
      A((function() {
        var r = p(), e = r[0], a = r[1];
        sr(E.j, e);
        s("r", [ a ]);
      }));
    }
    return E;
  };
  var Za = function getEnvironment() {
    if (!Wa) {
      Wa = Xa();
    }
    return Wa;
  };
  var Ga = function createEventContentChange(r, e, a) {
    var n = false;
    var t = a ? new WeakMap : false;
    var i = function destroy() {
      n = true;
    };
    var v = function updateElements(i) {
      if (t && a) {
        var v = a.map((function(e) {
          var a = e || [], n = a[0], t = a[1];
          var v = t && n ? (i || Pr)(n, r) : [];
          return [ v, t ];
        }));
        each(v, (function(a) {
          return each(a[0], (function(i) {
            var v = a[1];
            var o = t.get(i) || [];
            var u = r.contains(i);
            if (u && v) {
              var c = se(i, v, (function(r) {
                if (n) {
                  c();
                  t.delete(i);
                } else {
                  e(r);
                }
              }));
              t.set(i, L(o, c));
            } else {
              N(o);
              t.delete(i);
            }
          }));
        }));
      }
    };
    v();
    return [ i, v ];
  };
  var $a = function createDOMObserver(r, e, a, n) {
    var t = false;
    var i = n || {}, v = i.ir, o = i.vr, u = i.ur, c = i.cr, f = i.lr, l = i.sr;
    var s = Ga(r, (function() {
      return t && a(true);
    }), u), d = s[0], _ = s[1];
    var g = v || [];
    var h = o || [];
    var b = I(g, h);
    var m = function observerCallback(t, i) {
      if (!R(i)) {
        var v = f || _r;
        var o = l || _r;
        var u = [];
        var s = [];
        var d = false;
        var p = false;
        each(i, (function(a) {
          var t = a.attributeName, i = a.target, f = a.type, l = a.oldValue, _ = a.addedNodes, g = a.removedNodes;
          var b = f === "attributes";
          var m = f === "childList";
          var y = r === i;
          var S = b && t;
          var O = S && br(i, t || "");
          var C = w(O) ? O : null;
          var x = S && l !== C;
          var E = M(h, t) && x;
          if (e && (m || !y)) {
            var A = b && x;
            var T = A && c && zr(i, c);
            var H = T ? !v(i, t, l, C) : !b || A;
            var P = H && !o(a, !!T, r, n);
            each(_, (function(r) {
              return L(u, r);
            }));
            each(g, (function(r) {
              return L(u, r);
            }));
            p = p || P;
          }
          if (!e && y && x && !v(i, t, l, C)) {
            L(s, t);
            d = d || E;
          }
        }));
        _((function(r) {
          return F(u).reduce((function(e, a) {
            L(e, Pr(r, a));
            return zr(a, r) ? L(e, a) : e;
          }), []);
        }));
        if (e) {
          if (!t && p) {
            a(false);
          }
          return [ false ];
        }
        if (!R(s) || d) {
          var g = [ F(s), d ];
          if (!t) {
            a.apply(0, g);
          }
          return g;
        }
      }
    };
    var y = new p(vr(m, false));
    return [ function() {
      y.observe(r, {
        attributes: true,
        attributeOldValue: true,
        attributeFilter: b,
        subtree: e,
        childList: e,
        characterData: e
      });
      t = true;
      return function() {
        if (t) {
          d();
          y.disconnect();
          t = false;
        }
      };
    }, function() {
      if (t) {
        return m(true, y.takeRecords());
      }
    } ];
  };
  var Ja = null;
  var Ka = function createSizeObserver(r, a, n) {
    var t = n || {}, i = t.dr;
    var v = He(Ca);
    var o = e({
      i: false,
      o: true
    }), u = o[0];
    return function() {
      var e = [];
      var n = Yr('<div class="' + ea + '"><div class="' + na + '"></div></div>');
      var t = n[0];
      var o = t.firstChild;
      var c = function onSizeChangedCallbackProxy(r) {
        var e = x(r) && !R(r);
        var n = false;
        var t = false;
        if (e) {
          var i = r[0];
          var v = u(i.contentRect), o = v[0], c = v[2];
          var f = ce(o);
          t = fe(o, c);
          n = !t && !f;
        } else {
          t = r === true;
        }
        if (!n) {
          a({
            pr: true,
            dr: t
          });
        }
      };
      if (g) {
        if (!O(Ja)) {
          var f = new g(_r);
          f.observe(r, {
            get box() {
              Ja = true;
            }
          });
          Ja = Ja || false;
          f.disconnect();
        }
        var l = cr(c, {
          p: 0,
          _: 0
        });
        var s = function resizeObserverCallback(r) {
          return l(r);
        };
        var d = new g(s);
        d.observe(Ja ? r : o);
        L(e, [ function() {
          d.disconnect();
        }, !Ja && Nr(r, t) ]);
        if (Ja) {
          var p = new g(s);
          p.observe(r, {
            box: "border-box"
          });
          L(e, (function() {
            return p.disconnect();
          }));
        }
      } else if (v) {
        var _ = v(o, c, i), h = _[0], b = _[1];
        L(e, I([ Hr(t, aa), se(t, "animationstart", h), Nr(r, t) ], b));
      } else {
        return _r;
      }
      return vr(N, e);
    };
  };
  var Qa = function createTrinsicObserver(r, a) {
    var n;
    var t = function isHeightIntrinsic(r) {
      return r.h === 0 || r.isIntersecting || r.intersectionRatio > 0;
    };
    var i = Br(oa);
    var v = e({
      i: false
    }), o = v[0];
    var u = function triggerOnTrinsicChangedCallback(r, e) {
      if (r) {
        var n = o(t(r));
        var i = n[1];
        return i && !e && a(n) && [ n ];
      }
    };
    var c = function intersectionObserverCallback(r, e) {
      return u(e.pop(), r);
    };
    return [ function() {
      var e = [];
      if (_) {
        n = new _(vr(c, false), {
          root: r
        });
        n.observe(i);
        L(e, (function() {
          n.disconnect();
        }));
      } else {
        var a = function onSizeChanged() {
          var r = ne(i);
          u(r);
        };
        L(e, Ka(i, a)());
        a();
      }
      return vr(N, L(e, Nr(r, i)));
    }, function() {
      return n && c(true, n.takeRecords());
    } ];
  };
  var rn = function createObserversSetup(r, a, n, t) {
    var i;
    var v;
    var o;
    var u;
    var c;
    var f;
    var l;
    var s;
    var d = "[" + qe + "]";
    var p = "[" + Ye + "]";
    var _ = [ "id", "class", "style", "open", "wrap", "cols", "rows" ];
    var h = r._r, b = r.gr, m = r.F, y = r.hr, w = r.br, O = r.R, E = r.mr, A = r.yr, H = r.Sr, P = r.wr;
    var D = function getDirectionIsRTL(r) {
      return getStyles(r, "direction") === "rtl";
    };
    var z = function createDebouncedObservesUpdate() {
      var r;
      var e;
      var a;
      var n = cr(t, {
        p: function _debounceTiming() {
          return r;
        },
        _: function _maxDebounceTiming() {
          return e;
        },
        m: function _leading() {
          return a;
        },
        S: function _mergeParams(r, e) {
          var a = r[0];
          var n = e[0];
          return [ I(lr(a), lr(n)).reduce((function(r, e) {
            r[e] = a[e] || n[e];
            return r;
          }), {}) ];
        }
      });
      var i = function fn(t, i) {
        if (x(i)) {
          var v = i[0], o = i[1], u = i[2];
          r = v;
          e = o;
          a = u;
        } else if (S(i)) {
          r = i;
          e = false;
          a = false;
        } else {
          r = false;
          e = false;
          a = false;
        }
        n(t);
      };
      i.O = n.O;
      return i;
    };
    var M = {
      Or: false,
      B: D(h)
    };
    var L = Za();
    var k = He(Da);
    var V = e({
      v: nr,
      i: {
        w: 0,
        h: 0
      }
    }, (function() {
      var e = k && k.V(r, a, M, L, n).Z;
      var t = E && O;
      var i = !t && Er(b, qe, je);
      var v = !O && A(Ze);
      var o = v && he(y);
      var u = o && P();
      var c = H(Ge, i);
      var f = v && e && e();
      var l = ie(m);
      var s = ve(m);
      if (f) {
        f();
      }
      ge(y, o);
      if (u) {
        u();
      }
      if (i) {
        c();
      }
      return {
        w: l.w + s.w,
        h: l.h + s.h
      };
    })), R = V[0];
    var F = z();
    var N = function setDirection(r) {
      var e = D(h);
      sr(r, {
        Cr: s !== e
      });
      sr(M, {
        B: e
      });
      s = e;
    };
    var j = function onTrinsicChanged(r, e) {
      var a = r[0], n = r[1];
      var i = {
        Er: n
      };
      sr(M, {
        Or: a
      });
      if (!e) {
        t(i);
      }
      return i;
    };
    var U = function onSizeChanged(r) {
      var e = r.pr, a = r.dr;
      var n = a ? t : F;
      var i = {
        pr: e || a,
        dr: a
      };
      N(i);
      n(i, v);
    };
    var q = function onContentMutation(r, e) {
      var a = R(), n = a[1];
      var t = {
        Ar: n
      };
      N(t);
      if (n && !e) {
        F(t, r ? o : i);
      }
      return t;
    };
    var B = function onHostMutation(r, e, a) {
      var n = {
        Tr: e
      };
      N(n);
      if (e && !a) {
        F(n, i);
      }
      return n;
    };
    var Y = w ? Qa(b, j) : [], W = Y[0], X = Y[1];
    var Z = !O && Ka(b, U, {
      dr: true
    });
    var G = $a(b, false, B, {
      vr: _,
      ir: _
    }), $ = G[0], J = G[1];
    var K = O && g && new g((function(r) {
      var e = r[r.length - 1].contentRect;
      U({
        pr: true,
        dr: fe(e, l)
      });
      l = e;
    }));
    return [ function() {
      if (K) {
        K.observe(b);
      }
      var r = Z && Z();
      var e = W && W();
      var a = $();
      var n = L.J((function(r) {
        var e = R(), a = e[1];
        F({
          Hr: r,
          Ar: a,
          pr: E
        }, u);
      }));
      return function() {
        if (K) {
          K.disconnect();
        }
        if (r) {
          r();
        }
        if (e) {
          e();
        }
        if (f) {
          f();
        }
        a();
        n();
      };
    }, function(r) {
      var e = r.Pr, a = r.Dr, n = r.zr;
      var t = {};
      var l = e("update.ignoreMutation"), s = l[0];
      var g = e("update.attributes"), h = g[0], b = g[1];
      var y = e("update.elementEvents"), E = y[0], A = y[1];
      var H = e("update.debounce"), P = H[0], D = H[1];
      var z = A || b;
      var M = a || n;
      var L = function ignoreMutationFromOptions(r) {
        return C(s) && s(r);
      };
      if (z) {
        if (c) {
          c();
        }
        if (f) {
          f();
        }
        var k = $a(w || m, true, q, {
          ir: I(_, h || []),
          ur: E,
          cr: d,
          sr: function _ignoreContentChange(r, e) {
            var a = r.target, n = r.attributeName;
            var t = !e && n && !O ? Rr(a, d, p) : false;
            return t || !!kr(a, "." + ca) || !!L(r);
          }
        }), V = k[0], R = k[1];
        f = V();
        c = R;
      }
      if (D) {
        F.O();
        if (x(P) || S(P)) {
          i = P;
          v = false;
          o = Ra;
          u = Fa;
        } else if (T(P)) {
          i = P.mutation;
          v = P.resize;
          o = P.event;
          u = P.env;
        } else {
          i = false;
          v = false;
          o = false;
          u = false;
        }
      }
      if (M) {
        var U = J();
        var Y = X && X();
        var W = c && c();
        if (U) {
          sr(t, B(U[0], U[1], M));
        }
        if (Y) {
          sr(t, j(Y[0], M));
        }
        if (W) {
          sr(t, q(W[0], M));
        }
      }
      N(t);
      return t;
    }, M ];
  };
  var en = function resolveInitialization(r, e) {
    return C(e) ? e.apply(0, r) : e;
  };
  var an = function staticInitializationElement(r, e, a, n) {
    var t = b(n) ? a : n;
    var i = en(r, t);
    return i || e.apply(0, r);
  };
  var nn = function dynamicInitializationElement(r, e, a, n) {
    var t = b(n) ? a : n;
    var i = en(r, t);
    return !!i && (H(i) ? i : e.apply(0, r));
  };
  var tn = function cancelInitialization(r, e) {
    var a = e || {}, n = a.nativeScrollbarsOverlaid, t = a.body;
    var i = Za(), v = i.k, o = i.N, u = i.K;
    var c = u().cancel, f = c.nativeScrollbarsOverlaid, l = c.body;
    var s = n != null ? n : f;
    var d = b(t) ? l : t;
    var p = (v.x || v.y) && s;
    var _ = r && (m(d) ? !o : d);
    return !!p || !!_;
  };
  var vn = function createScrollbarsSetupElements(r, e, a, n) {
    var t = "--os-viewport-percent";
    var i = "--os-scroll-percent";
    var v = "--os-scroll-direction";
    var o = Za(), u = o.K;
    var c = u(), f = c.scrollbars;
    var l = f.slot;
    var s = e._r, d = e.gr, p = e.F, _ = e.Mr, g = e.hr, b = e.mr, m = e.R;
    var y = _ ? {} : r, S = y.scrollbars;
    var w = S || {}, C = w.slot;
    var x = [];
    var E = [];
    var A = [];
    var T = nn([ s, d, p ], (function() {
      return m && b ? s : d;
    }), l, C);
    var H = function initScrollTimeline(r) {
      if (h) {
        var e = null;
        var n = [];
        var t = new h({
          source: g,
          axis: r
        });
        var i = function cancelAnimation() {
          if (e) {
            e.cancel();
          }
          e = null;
        };
        var v = function _setScrollPercentAnimation(v) {
          var o = a.Ir;
          var u = ye(o)[r];
          var c = r === "x";
          var f = [ Kr(0, c), Kr("calc(-100% + 100cq" + (c ? "w" : "h") + ")", c) ];
          var l = u ? f : f.reverse();
          if (n[0] === l[0] && n[1] === l[1]) {
            return i;
          }
          n = l;
          i();
          e = v.Lr.animate({
            clear: [ "left" ],
            transform: l
          }, {
            timeline: t
          });
          return i;
        };
        return {
          kr: v
        };
      }
    };
    var P = {
      x: H("x"),
      y: H("y")
    };
    var D = function getViewportPercent() {
      var r = a.Vr, e = a.Rr;
      var n = function getAxisValue(r, e) {
        return gr(0, 1, r / (r + e) || 0);
      };
      return {
        x: n(e.x, r.x),
        y: n(e.y, r.y)
      };
    };
    var z = function scrollbarStructureAddRemoveClass(r, e, a) {
      var n = a ? Hr : Tr;
      each(r, (function(r) {
        n(r.Fr, e);
      }));
    };
    var M = function scrollbarStyle(r, e) {
      each(r, (function(r) {
        var a = e(r), n = a[0], t = a[1];
        setStyles(n, t);
      }));
    };
    var I = function scrollbarsAddRemoveClass(r, e, a) {
      var n = O(a);
      var t = n ? a : true;
      var i = n ? !a : true;
      if (t) {
        z(E, r, e);
      }
      if (i) {
        z(A, r, e);
      }
    };
    var k = function refreshScrollbarsHandleLength() {
      var r = D();
      var e = function createScrollbarStyleFn(r) {
        return function(e) {
          var a;
          return [ e.Fr, (a = {}, a[t] = Gr(r) + "", a) ];
        };
      };
      M(E, e(r.x));
      M(A, e(r.y));
    };
    var V = function refreshScrollbarsHandleOffset() {
      if (!h) {
        var r = a.Ir;
        var e = Se(r, he(g));
        var n = function createScrollbarStyleFn(r) {
          return function(e) {
            var a;
            return [ e.Fr, (a = {}, a[i] = Gr(r) + "", a) ];
          };
        };
        M(E, n(e.x));
        M(A, n(e.y));
      }
    };
    var R = function refreshScrollbarsScrollCoordinates() {
      var r = a.Ir;
      var e = ye(r);
      var n = function createScrollbarStyleFn(r) {
        return function(e) {
          var a;
          return [ e.Fr, (a = {}, a[v] = r ? "0" : "1", a) ];
        };
      };
      M(E, n(e.x));
      M(A, n(e.y));
      if (h) {
        E.forEach(P.x.kr);
        A.forEach(P.y.kr);
      }
    };
    var F = function refreshScrollbarsScrollbarOffset() {
      if (m && !b) {
        var r = a.Vr, e = a.Ir;
        var n = ye(e);
        var t = Se(e, he(g));
        var i = function styleScrollbarPosition(e) {
          var a = e.Fr;
          var i = Lr(a) === p && a;
          var v = function getTranslateValue(r, e, a) {
            var n = e * r;
            return $r(a ? n : -n);
          };
          return [ i, i && {
            transform: Kr({
              x: v(t.x, r.x, n.x),
              y: v(t.y, r.y, n.y)
            })
          } ];
        };
        M(E, i);
        M(A, i);
      }
    };
    var j = function generateScrollbarDOM(r) {
      var e = r ? "x" : "y";
      var a = r ? la : sa;
      var t = Br(ca + " " + a);
      var i = Br(da);
      var v = Br(pa);
      var o = {
        Fr: t,
        Nr: i,
        Lr: v
      };
      var u = P[e];
      L(r ? E : A, o);
      L(x, [ Nr(t, i), Nr(i, v), vr(Fr, t), u && u.kr(o), n(o, I, r) ]);
      return o;
    };
    var U = vr(j, true);
    var q = vr(j, false);
    var B = function appendElements() {
      Nr(T, E[0].Fr);
      Nr(T, A[0].Fr);
      return vr(N, x);
    };
    U();
    q();
    return [ {
      jr: k,
      Ur: V,
      qr: R,
      Br: F,
      Yr: I,
      Wr: {
        Xr: E,
        Zr: U,
        Gr: vr(M, E)
      },
      $r: {
        Xr: A,
        Zr: q,
        Gr: vr(M, A)
      }
    }, B ];
  };
  var on = function createScrollbarsSetupEvents(r, e, a, n) {
    return function(t, i, u) {
      var c = e.gr, f = e.F, s = e.R, d = e.hr, p = e.Jr, _ = e.wr;
      var g = t.Fr, h = t.Nr, b = t.Lr;
      var m = or(333), y = m[0], S = m[1];
      var w = or(444), O = w[0], x = w[1];
      var E = function scrollOffsetElementScrollBy(r) {
        if (C(d.scrollBy)) {
          d.scrollBy({
            behavior: "smooth",
            left: r.x,
            top: r.y
          });
        }
      };
      var A = function createInteractiveScrollEvents() {
        var e = "pointerup pointercancel lostpointercapture";
        var n = "client" + (u ? "X" : "Y");
        var t = u ? $ : J;
        var i = u ? "left" : "top";
        var c = u ? "w" : "h";
        var f = u ? "x" : "y";
        var l = function createRelativeHandleMove(r, e) {
          return function(n) {
            var t;
            var i = a.Vr;
            var v = ne(h)[c] - ne(b)[c];
            var o = e * n / v;
            var u = o * i[f];
            ge(d, (t = {}, t[f] = r + u, t));
          };
        };
        var s = [];
        return se(h, "pointerdown", (function(a) {
          var u = kr(a.target, "." + pa) === b;
          var g = u ? b : h;
          var m = r.scrollbars;
          var y = m[u ? "dragScroll" : "clickScroll"];
          var S = a.button, w = a.isPrimary, C = a.pointerType;
          var A = m.pointers;
          var T = S === 0 && w && y && (A || []).includes(C);
          if (T) {
            N(s);
            x();
            var H = !u && (a.shiftKey || y === "instant");
            var P = vr(oe, b);
            var D = vr(oe, h);
            var z = function getHandleOffset(r, e) {
              return (r || P())[i] - (e || D())[i];
            };
            var M = v(oe(d)[t]) / ne(d)[c] || 1;
            var I = l(he(d)[f], 1 / M);
            var k = a[n];
            var V = P();
            var R = D();
            var F = V[t];
            var j = z(V, R) + F / 2;
            var U = k - R[i];
            var q = u ? 0 : U - j;
            var B = function releasePointerCapture(r) {
              N(X);
              g.releasePointerCapture(r.pointerId);
            };
            var Y = u || H;
            var W = _();
            var X = [ se(p, e, B), se(p, "selectstart", (function(r) {
              return pe(r);
            }), {
              H: false
            }), se(h, e, B), Y && se(h, "pointermove", (function(r) {
              return I(q + (r[n] - k));
            })), Y && function() {
              var r = he(d);
              W();
              var e = he(d);
              var a = {
                x: e.x - r.x,
                y: e.y - r.y
              };
              if (o(a.x) > 3 || o(a.y) > 3) {
                _();
                ge(d, r);
                E(a);
                O(W);
              }
            } ];
            g.setPointerCapture(a.pointerId);
            if (H) {
              I(q);
            } else if (!u) {
              var Z = He(Ma);
              if (Z) {
                var G = Z(I, q, F, (function(r) {
                  if (r) {
                    W();
                  } else {
                    L(X, W);
                  }
                }));
                L(X, G);
                L(s, vr(G, true));
              }
            }
          }
        }));
      };
      var T = true;
      return vr(N, [ se(b, "pointermove pointerleave", n), se(g, "pointerenter", (function() {
        i(ha, true);
      })), se(g, "pointerleave pointercancel", (function() {
        i(ha, false);
      })), !s && se(g, "mousedown", (function() {
        var r = Vr();
        if (mr(r, Ye) || mr(r, qe) || r === document.body) {
          l(vr(we, f), 25);
        }
      })), se(g, "wheel", (function(r) {
        var e = r.deltaX, a = r.deltaY, n = r.deltaMode;
        if (T && n === 0 && Lr(g) === c) {
          E({
            x: e,
            y: a
          });
        }
        T = false;
        i(Sa, true);
        y((function() {
          T = true;
          i(Sa);
        }));
        pe(r);
      }), {
        H: false,
        P: true
      }), se(g, "pointerdown", (function() {
        var r = se(p, "click", (function(r) {
          e();
          _e(r);
        }), {
          D: true,
          P: true,
          H: false
        });
        var e = se(p, "pointerup pointercancel", (function() {
          e();
          setTimeout(r, 150);
        }), {
          P: true,
          H: true
        });
      }), {
        P: true,
        H: true
      }), A(), S, x ]);
    };
  };
  var un = function createScrollbarsSetup(r, e, a, n, t, i) {
    var v;
    var o;
    var u;
    var c;
    var f;
    var l = _r;
    var s = 0;
    var d = [ "mouse", "pen" ];
    var p = function isHoverablePointerType(r) {
      return d.includes(r.pointerType);
    };
    var _ = or(), g = _[0], h = _[1];
    var b = or(100), m = b[0], y = b[1];
    var S = or(100), w = S[0], O = S[1];
    var C = or((function() {
      return s;
    })), x = C[0], E = C[1];
    var A = vn(r, t, n, on(e, t, n, (function(r) {
      return p(r) && j();
    }))), T = A[0], H = A[1];
    var P = t.gr, D = t.Kr, z = t.mr;
    var M = T.Yr, I = T.jr, k = T.Ur, V = T.qr, R = T.Br;
    var F = function manageScrollbarsAutoHide(r, e) {
      E();
      if (r) {
        M(ya);
      } else {
        var a = vr(M, ya, true);
        if (s > 0 && !e) {
          x(a);
        } else {
          a();
        }
      }
    };
    var j = function manageScrollbarsAutoHideInstantInteraction() {
      if (u ? !v : !c) {
        F(true);
        m((function() {
          F(false);
        }));
      }
    };
    var U = function manageAutoHideSuspension(r) {
      M(ma, r, true);
      M(ma, r, false);
    };
    var q = function onHostMouseEnter(r) {
      if (p(r)) {
        v = u;
        if (u) {
          F(true);
        }
      }
    };
    var B = [ E, y, O, h, function() {
      return l();
    }, se(P, "pointerover", q, {
      D: true
    }), se(P, "pointerenter", q), se(P, "pointerleave", (function(r) {
      if (p(r)) {
        v = false;
        if (u) {
          F(false);
        }
      }
    })), se(P, "pointermove", (function(r) {
      if (p(r) && o) {
        j();
      }
    })), se(D, "scroll", (function(r) {
      g((function() {
        k();
        j();
      }));
      i(r);
      R();
    })) ];
    var Y = He(Da);
    return [ function() {
      return vr(N, L(B, H()));
    }, function(r) {
      var e = r.Pr, t = r.zr, i = r.Qr, v = r.re;
      var d = v || {}, p = d.ee, _ = d.ae, g = d.ne, h = d.te;
      var b = i || {}, m = b.Cr, y = b.dr;
      var S = a.B;
      var O = Za(), C = O.k, x = O.N;
      var E = n.ie, A = n.W;
      var T = e("showNativeOverlaidScrollbars"), H = T[0], P = T[1];
      var L = e("scrollbars.theme"), N = L[0], j = L[1];
      var q = e("scrollbars.visibility"), B = q[0], W = q[1];
      var X = e("scrollbars.autoHide"), Z = X[0], G = X[1];
      var $ = e("scrollbars.autoHideSuspend"), J = $[0], Q = $[1];
      var er = e("scrollbars.autoHideDelay"), ar = er[0];
      var nr = e("scrollbars.dragScroll"), tr = nr[0], ir = nr[1];
      var or = e("scrollbars.clickScroll"), ur = or[0], cr = or[1];
      var fr = e("overflow"), lr = fr[0], sr = fr[1];
      var dr = y && !t;
      var pr = A.x || A.y;
      var _r = p || _ || h || m || t;
      var gr = g || W || sr;
      var hr = H && C.x && C.y;
      var br = !x && !Y;
      var mr = hr || br;
      var yr = function setScrollbarVisibility(r, e, a) {
        var n = r.includes(rr) && (B === K || B === "auto" && e === rr);
        M(_a, n, a);
        return n;
      };
      s = ar;
      if (dr) {
        if (J && pr) {
          U(false);
          l();
          w((function() {
            l = se(D, "scroll", vr(U, true), {
              D: true
            });
          }));
        } else {
          U(true);
        }
      }
      if (P || br) {
        M(ua, mr);
      }
      if (j) {
        M(f);
        M(N, true);
        f = N;
      }
      if (Q && !J) {
        U(true);
      }
      if (G) {
        o = Z === "move";
        u = Z === "leave";
        c = Z === "never";
        F(c, true);
      }
      if (ir) {
        M(Oa, tr);
      }
      if (cr) {
        M(wa, !!ur);
      }
      if (gr) {
        var Sr = yr(lr.x, E.x, true);
        var wr = yr(lr.y, E.y, false);
        var Or = Sr && wr;
        M(ga, !Or);
      }
      if (_r) {
        k();
        I();
        R();
        if (h) {
          V();
        }
        M(ba, !A.x, true);
        M(ba, !A.y, false);
        M(fa, S && !z);
      }
    }, {}, T ];
  };
  var cn = function createStructureSetupElements(r) {
    var e = Za();
    var a = e.K, t = e.N;
    var i = a(), v = i.elements;
    var o = v.padding, u = v.viewport, c = v.content;
    var f = H(r);
    var l = f ? {} : r;
    var s = l.elements;
    var d = s || {}, p = d.padding, _ = d.viewport, g = d.content;
    var h = f ? r : l.target;
    var b = Mr(h);
    var m = h.ownerDocument;
    var y = m.documentElement;
    var S = function getDocumentWindow() {
      return m.defaultView || n;
    };
    var w = vr(an, [ h ]);
    var O = vr(nn, [ h ]);
    var C = vr(Br, "");
    var x = vr(w, C, u);
    var E = vr(O, C, c);
    var A = function elementHasOverflow(r) {
      var e = ne(r);
      var a = ie(r);
      var n = getStyles(r, Z);
      var t = getStyles(r, G);
      return a.w - e.w > 0 && !Aa(n) || a.h - e.h > 0 && !Aa(t);
    };
    var T = x(_);
    var P = T === h;
    var D = P && b;
    var z = !P && E(g);
    var I = !P && T === z;
    var k = D ? y : T;
    var V = D ? k : h;
    var R = !P && O(C, o, p);
    var F = !I && z;
    var j = [ F, k, R, V ].map((function(r) {
      return H(r) && !Lr(r) && r;
    }));
    var U = function elementIsGenerated(r) {
      return r && M(j, r);
    };
    var q = !U(k) && A(k) ? k : h;
    var B = D ? y : k;
    var Y = D ? m : k;
    var W = {
      _r: h,
      gr: V,
      F: k,
      ve: R,
      br: F,
      hr: B,
      Kr: Y,
      oe: b ? y : q,
      Jr: m,
      mr: b,
      Mr: f,
      R: P,
      ue: S,
      yr: function _viewportHasClass(r) {
        return Er(k, Ye, r);
      },
      Sr: function _viewportAddRemoveClass(r, e) {
        return xr(k, Ye, r, e);
      },
      wr: function _removeScrollObscuringStyles() {
        return xr(B, Ye, $e, true);
      }
    };
    var X = W._r, $ = W.gr, J = W.ve, K = W.F, Q = W.br;
    var rr = [ function() {
      Sr($, [ qe, Ne ]);
      Sr(X, Ne);
      if (b) {
        Sr(y, [ Ne, qe ]);
      }
    } ];
    var er = Ir([ Q, K, J, $, X ].find((function(r) {
      return r && !U(r);
    })));
    var ar = D ? X : Q || K;
    var nr = vr(N, rr);
    var tr = function appendElements() {
      var r = S();
      var e = Vr();
      var a = function unwrap(r) {
        Nr(Lr(r), Ir(r));
        Fr(r);
      };
      var n = function prepareWrapUnwrapFocus(r) {
        return se(r, "focusin focusout focus blur", _e, {
          P: true,
          H: false
        });
      };
      var i = "tabindex";
      var v = br(K, i);
      var o = n(e);
      yr($, qe, P ? "" : Be);
      yr(J, Qe, "");
      yr(K, Ye, "");
      yr(Q, ra, "");
      if (!P) {
        yr(K, i, v || "-1");
        if (b) {
          yr(y, Ue, "");
        }
      }
      Nr(ar, er);
      Nr($, J);
      Nr(J || $, !P && K);
      Nr(K, Q);
      L(rr, [ o, function() {
        var r = Vr();
        var e = U(K);
        var t = e && r === K ? X : r;
        var o = n(t);
        Sr(J, Qe);
        Sr(Q, ra);
        Sr(K, Ye);
        if (b) {
          Sr(y, Ue);
        }
        if (v) {
          yr(K, i, v);
        } else {
          Sr(K, i);
        }
        if (U(Q)) {
          a(Q);
        }
        if (e) {
          a(K);
        }
        if (U(J)) {
          a(J);
        }
        we(t);
        o();
      } ]);
      if (t && !P) {
        Cr(K, Ye, Je);
        L(rr, vr(Sr, K, Ye));
      }
      we(!P && b && e === X && r.top === r ? K : e);
      o();
      er = 0;
      return nr;
    };
    return [ W, tr, nr ];
  };
  var ln = function createTrinsicUpdateSegment(r) {
    var e = r.br;
    return function(r) {
      var a = r.Qr, n = r.ce, t = r.zr;
      var i = a || {}, v = i.Er;
      var o = n.Or;
      var u = e && (v || t);
      if (u) {
        var c;
        setStyles(e, (c = {}, c[J] = o && "100%", c));
      }
    };
  };
  var sn = function createPaddingUpdateSegment(r, a) {
    var n = r.gr, t = r.ve, i = r.F, v = r.R;
    var o = e({
      v: ir,
      i: Jr()
    }, vr(Jr, n, "padding", "")), u = o[0], c = o[1];
    return function(r) {
      var e = r.Pr, n = r.Qr, o = r.ce, f = r.zr;
      var l = c(f), s = l[0], d = l[1];
      var p = Za(), _ = p.N;
      var g = n || {}, h = g.pr, b = g.Ar, m = g.Cr;
      var y = o.B;
      var S = e("paddingAbsolute"), w = S[0], O = S[1];
      var C = f || b;
      if (h || d || C) {
        var x = u(f);
        s = x[0];
        d = x[1];
      }
      var E = !v && (O || m || d);
      if (E) {
        var A, T;
        var H = !w || !t && !_;
        var P = s.r + s.l;
        var D = s.t + s.b;
        var z = (A = {}, A[W] = H && !y ? -P : 0, A[X] = H ? -D : 0, A[Y] = H && y ? -P : 0, 
        A.top = H ? -s.t : 0, A.right = H ? y ? -s.r : "auto" : 0, A.left = H ? y ? "auto" : -s.l : 0, 
        A[$] = H && "calc(100% + " + P + "px)", A);
        var M = (T = {}, T[j] = H ? s.t : 0, T[U] = H ? s.r : 0, T[B] = H ? s.b : 0, T[q] = H ? s.l : 0, 
        T);
        setStyles(t || i, z);
        setStyles(i, M);
        sr(a, {
          ve: s,
          fe: !H,
          Y: t ? M : sr({}, z, M)
        });
      }
      return {
        le: E
      };
    };
  };
  var dn = function createOverflowUpdateSegment(r, a) {
    var i = Za();
    var v = r.gr, u = r.ve, c = r.F, l = r.R, s = r.Kr, d = r.hr, p = r.mr, _ = r.Sr, g = r.ue;
    var h = i.N;
    var b = p && l;
    var m = vr(t, 0);
    var y = {
      display: function display() {
        return false;
      },
      direction: function direction(r) {
        return r !== "ltr";
      },
      flexDirection: function flexDirection(r) {
        return r.endsWith("-reverse");
      },
      writingMode: function writingMode(r) {
        return r !== "horizontal-tb";
      }
    };
    var S = lr(y);
    var O = {
      v: nr,
      i: {
        w: 0,
        h: 0
      }
    };
    var C = {
      v: tr,
      i: {}
    };
    var x = function setMeasuringMode(r) {
      _(Ge, !b && r);
    };
    var E = function getFlowDirectionStyles() {
      return getStyles(c, S);
    };
    var A = function getMeasuredScrollCoordinates(r, e) {
      var a = !lr(r).length;
      var n = !e && S.some((function(e) {
        var a = r[e];
        return w(a) && y[e](a);
      }));
      var t = a && !n;
      if (t || !ue(c)) {
        return {
          M: {
            x: 0,
            y: 0
          },
          I: {
            x: 1,
            y: 1
          }
        };
      }
      x(true);
      var i = he(d);
      var v = se(s, rr, (function(r) {
        var e = he(d);
        if (r.isTrusted && e.x === i.x && e.y === i.y) {
          de(r);
        }
      }), {
        P: true,
        D: true
      });
      var u = _(Ke, true);
      ge(d, {
        x: 0,
        y: 0
      });
      u();
      var l = he(d);
      var p = ie(d);
      ge(d, {
        x: p.w,
        y: p.h
      });
      var g = he(d);
      var h = {
        x: g.x - l.x,
        y: g.y - l.y
      };
      ge(d, {
        x: -p.w,
        y: -p.h
      });
      var b = he(d);
      var m = {
        x: b.x - l.x,
        y: b.y - l.y
      };
      var O = {
        x: o(h.x) >= o(m.x) ? g.x : b.x,
        y: o(h.y) >= o(m.y) ? g.y : b.y
      };
      ge(d, i);
      f((function() {
        return v();
      }));
      return {
        M: l,
        I: O
      };
    };
    var T = function getOverflowAmount(r, e) {
      var a = n.devicePixelRatio % 1 !== 0 ? 1 : 0;
      var t = {
        w: m(r.w - e.w),
        h: m(r.h - e.h)
      };
      return {
        w: t.w > a ? t.w : 0,
        h: t.h > a ? t.h : 0
      };
    };
    var H = function getViewportOverflowStyle(r, e) {
      var a = function getAxisOverflowStyle(r, e, a, n) {
        var t = r === K ? Q : Ta(r);
        var i = Aa(r);
        var v = Aa(a);
        if (!e && !n) {
          return Q;
        }
        if (i && v) {
          return K;
        }
        if (i) {
          var o = e ? K : Q;
          return e && n ? t : o;
        }
        var u = v && n ? K : Q;
        return e ? t : u;
      };
      return {
        x: a(e.x, r.x, e.y, r.y),
        y: a(e.y, r.y, e.x, r.x)
      };
    };
    var P = function setViewportOverflowStyle(r) {
      var e = function createAllOverflowStyleClassNames(r) {
        return [ K, Q, rr ].map((function(e) {
          return cr(Ha(e), r);
        }));
      };
      var a = e(true).concat(e()).join(" ");
      _(a);
      _(lr(r).map((function(e) {
        return cr(r[e], e === "x");
      })).join(" "), true);
    };
    var D = e(O, vr(ve, c)), z = D[0], M = D[1];
    var L = e(O, vr(ie, c)), k = L[0], V = L[1];
    var R = e(O), N = R[0], j = R[1];
    var U = e(C), q = U[0];
    var B = e(O), Y = B[0], W = B[1];
    var X = e(C), Z = X[0];
    var G = e({
      v: function _equal(r, e) {
        return ar(r, e, F(I(lr(r), lr(e))));
      },
      i: {}
    }), $ = G[0];
    var J = e({
      v: function _equal(r, e) {
        return tr(r.M, e.M) && tr(r.I, e.I);
      },
      i: be()
    }), ir = J[0], or = J[1];
    var ur = He(Da);
    var cr = function createViewportOverflowStyleClassName(r, e) {
      var a = e ? We : Xe;
      return "" + a + er(r);
    };
    return function(e, n) {
      var t = e.Pr, o = e.Qr, f = e.ce, l = e.zr;
      var s = n.le;
      var d = o || {}, p = d.pr, y = d.Tr, S = d.Ar, w = d.Cr, O = d.dr, C = d.Hr;
      var D = ur && ur.V(r, a, f, i, t);
      var I = D || {}, L = I.X, R = I.Z, F = I.G;
      var U = Ea(t, i), B = U[0], X = U[1];
      var G = t("overflow"), J = G[0], K = G[1];
      var Q = Aa(J.x);
      var rr = Aa(J.y);
      var er = p || s || S || w || C || X;
      var ar = M(l);
      var nr = V(l);
      var tr = j(l);
      var vr = W(l);
      if (X && h) {
        _(Je, !B);
      }
      if (er) {
        if (Er(v, qe, je)) {
          x(true);
        }
        var cr = R && R();
        var fr = ar = z(l), lr = fr[0];
        var dr = nr = k(l), pr = dr[0];
        var _r = te(c);
        var gr = b && ae(g());
        var hr = {
          w: m(pr.w + lr.w),
          h: m(pr.h + lr.h)
        };
        var br = {
          w: m((gr ? gr.w : _r.w + m(_r.w - pr.w)) + lr.w),
          h: m((gr ? gr.h : _r.h + m(_r.h - pr.h)) + lr.h)
        };
        if (cr) {
          cr();
        }
        vr = Y(br);
        tr = N(T(hr, br), l);
      }
      var mr = vr, yr = mr[0], Sr = mr[1];
      var wr = tr, Or = wr[0], Cr = wr[1];
      var Ar = nr, Tr = Ar[0], Hr = Ar[1];
      var Pr = ar, Dr = Pr[0], zr = Pr[1];
      var Mr = q({
        x: Or.w > 0,
        y: Or.h > 0
      }), Ir = Mr[0], Lr = Mr[1];
      var kr = Q && rr && (Ir.x || Ir.y) || Q && Ir.x && !Ir.y || rr && Ir.y && !Ir.x;
      var Vr = s || w || C || zr || Hr || Sr || Cr || K || X || er || y && b;
      var Rr = t("update.flowDirectionStyles"), Fr = Rr[0];
      var Nr = $(Fr ? Fr(c) : E(), l), jr = Nr[0], Ur = Nr[1];
      var qr = w || O || Ur || Lr || l;
      var Br = qr ? ir(A(jr, !!Fr), l) : or(), Yr = Br[0], Wr = Br[1];
      var Xr = H(Ir, J);
      x(false);
      if (Vr) {
        P(Xr);
        Xr = Pa(c, Ir);
        if (F && L) {
          L(Xr, Tr, Dr);
          setStyles(c, F(Xr));
        }
      }
      var Zr = Z(Xr), Gr = Zr[0], $r = Zr[1];
      xr(v, qe, je, kr);
      xr(u, Qe, je, kr);
      sr(a, {
        ie: Gr,
        Rr: {
          x: yr.w,
          y: yr.h
        },
        Vr: {
          x: Or.w,
          y: Or.h
        },
        W: Ir,
        Ir: me(Yr, Or)
      });
      return {
        ne: $r,
        ee: Sr,
        ae: Cr,
        te: Wr || Cr
      };
    };
  };
  var pn = function createStructureSetup(r) {
    var e;
    var a = cn(r), n = a[0], t = a[1], i = a[2];
    var v = {
      ve: {
        t: 0,
        r: 0,
        b: 0,
        l: 0
      },
      fe: false,
      Y: (e = {}, e[W] = 0, e[X] = 0, e[Y] = 0, e[j] = 0, e[U] = 0, e[B] = 0, e[q] = 0, 
      e),
      Rr: {
        x: 0,
        y: 0
      },
      Vr: {
        x: 0,
        y: 0
      },
      ie: {
        x: Q,
        y: Q
      },
      W: {
        x: false,
        y: false
      },
      Ir: be()
    };
    var o = n._r, u = n.hr, c = n.R, f = n.wr;
    var l = Za(), s = l.N, d = l.k;
    var p = !s && (d.x || d.y);
    var _ = [ ln(n), sn(n, v), dn(n, v) ];
    return [ t, function(r) {
      var e = {};
      var a = p;
      var n = a && he(u);
      var t = n && f();
      each(_, (function(a) {
        sr(e, a(r, e) || {});
      }));
      ge(u, n);
      if (t) {
        t();
      }
      if (!c) {
        ge(o, 0);
      }
      return e;
    }, v, n, i ];
  };
  var _n = function createSetups(r, e, a, n, t) {
    var i = false;
    var v = Ua(e, {});
    var o = pn(r), u = o[0], c = o[1], f = o[2], l = o[3], s = o[4];
    var d = rn(l, f, v, (function(r) {
      w({}, r);
    })), p = d[0], _ = d[1], g = d[2];
    var h = un(r, e, g, f, l, t), b = h[0], m = h[1], y = h[3];
    var S = function updateHintsAreTruthy(r) {
      return lr(r).some((function(e) {
        return !!r[e];
      }));
    };
    var w = function update(r, t) {
      if (a()) {
        return false;
      }
      var v = r.se, o = r.zr, u = r.Dr, f = r.de;
      var l = v || {};
      var s = !!o || !i;
      var d = {
        Pr: Ua(e, l, s),
        se: l,
        zr: s
      };
      if (f) {
        m(d);
        return false;
      }
      var p = t || _(sr({}, d, {
        Dr: u
      }));
      var h = c(sr({}, d, {
        ce: g,
        Qr: p
      }));
      m(sr({}, d, {
        Qr: p,
        re: h
      }));
      var b = S(p);
      var y = S(h);
      var w = b || y || !pr(l) || s;
      i = true;
      if (w) {
        n(r, {
          Qr: p,
          re: h
        });
      }
      return w;
    };
    return [ function() {
      var r = l.oe, e = l.hr, a = l.wr;
      var n = he(r);
      var t = [ p(), u(), b() ];
      var i = a();
      ge(e, n);
      i();
      return vr(N, t);
    }, w, function() {
      return {
        pe: g,
        _e: f
      };
    }, {
      ge: l,
      he: y
    }, s ];
  };
  var gn = new WeakMap;
  var hn = function addInstance(r, e) {
    gn.set(r, e);
  };
  var bn = function removeInstance(r) {
    gn.delete(r);
  };
  var mn = function getInstance(r) {
    return gn.get(r);
  };
  var yn = function OverlayScrollbars(r, e, a) {
    var n = Za(), t = n.er;
    var i = H(r);
    var v = i ? r : r.target;
    var o = mn(v);
    if (e && !o) {
      var u = false;
      var c = [];
      var f = {};
      var l = function validateOptions(r) {
        var e = dr(r);
        var a = He(ke);
        return a ? a(e, true) : e;
      };
      var s = sr({}, t(), l(e));
      var d = Ce(), p = d[0], _ = d[1], g = d[2];
      var h = Ce(a), b = h[0], m = h[1], y = h[2];
      var S = function triggerEvent(r, e) {
        y(r, e);
        g(r, e);
      };
      var w = _n(r, s, (function() {
        return u;
      }), (function(r, e) {
        var a = r.se, n = r.zr;
        var t = e.Qr, i = e.re;
        var v = t.pr, o = t.Cr, u = t.Er, c = t.Ar, f = t.Tr, l = t.dr;
        var s = i.ee, d = i.ae, p = i.ne, _ = i.te;
        S("updated", [ P, {
          updateHints: {
            sizeChanged: !!v,
            directionChanged: !!o,
            heightIntrinsicChanged: !!u,
            overflowEdgeChanged: !!s,
            overflowAmountChanged: !!d,
            overflowStyleChanged: !!p,
            scrollCoordinatesChanged: !!_,
            contentMutation: !!c,
            hostMutation: !!f,
            appear: !!l
          },
          changedOptions: a || {},
          force: !!n
        } ]);
      }), (function(r) {
        return S("scroll", [ P, r ]);
      })), O = w[0], C = w[1], x = w[2], E = w[3], A = w[4];
      var T = function destroy(r) {
        bn(v);
        N(c);
        u = true;
        S("destroyed", [ P, r ]);
        _();
        m();
      };
      var P = {
        options: function options(r, e) {
          if (r) {
            var a = e ? t() : {};
            var n = ja(s, sr(a, l(r)));
            if (!pr(n)) {
              sr(s, n);
              C({
                se: n
              });
            }
          }
          return sr({}, s);
        },
        on: b,
        off: function off(r, e) {
          if (r && e) {
            m(r, e);
          }
        },
        state: function state() {
          var r = x(), e = r.pe, a = r._e;
          var n = e.B;
          var t = a.Rr, i = a.Vr, v = a.ie, o = a.W, c = a.ve, f = a.fe, l = a.Ir;
          return sr({}, {
            overflowEdge: t,
            overflowAmount: i,
            overflowStyle: v,
            hasOverflow: o,
            scrollCoordinates: {
              start: l.M,
              end: l.I
            },
            padding: c,
            paddingAbsolute: f,
            directionRTL: n,
            destroyed: u
          });
        },
        elements: function elements() {
          var r = E.ge, e = r._r, a = r.gr, n = r.ve, t = r.F, i = r.br, v = r.hr, o = r.Kr;
          var u = E.he, c = u.Wr, f = u.$r;
          var l = function translateScrollbarStructure(r) {
            var e = r.Lr, a = r.Nr, n = r.Fr;
            return {
              scrollbar: n,
              track: a,
              handle: e
            };
          };
          var s = function translateScrollbarsSetupElement(r) {
            var e = r.Xr, a = r.Zr;
            var n = l(e[0]);
            return sr({}, n, {
              clone: function clone() {
                var r = l(a());
                C({
                  de: true
                });
                return r;
              }
            });
          };
          return sr({}, {
            target: e,
            host: a,
            padding: n || t,
            viewport: t,
            content: i || t,
            scrollOffsetElement: v,
            scrollEventElement: o,
            scrollbarHorizontal: s(c),
            scrollbarVertical: s(f)
          });
        },
        update: function update(r) {
          return C({
            zr: r,
            Dr: true
          });
        },
        destroy: vr(T, false),
        plugin: function plugin(r) {
          return f[lr(r)[0]];
        }
      };
      L(c, [ A ]);
      hn(v, P);
      Te(xe, yn, [ P, p, f ]);
      if (tn(E.ge.mr, !i && r.cancel)) {
        T(true);
        return P;
      }
      L(c, O());
      S("initialized", [ P ]);
      P.update();
      return P;
    }
    return o;
  };
  yn.plugin = function(r) {
    var e = x(r);
    var a = e ? r : [ r ];
    var n = a.map((function(r) {
      return Te(r, yn)[0];
    }));
    Ae(a);
    return e ? n : n[0];
  };
  yn.valid = function(r) {
    var e = r && r.elements;
    var a = C(e) && e();
    return T(a) && !!mn(a.target);
  };
  yn.env = function() {
    var r = Za(), e = r.j, a = r.k, n = r.N, t = r.$, i = r.nr, v = r.tr, o = r.K, u = r.rr, c = r.er, f = r.ar;
    return sr({}, {
      scrollbarsSize: e,
      scrollbarsOverlaid: a,
      scrollbarsHiding: n,
      scrollTimeline: t,
      staticDefaultInitialization: i,
      staticDefaultOptions: v,
      getDefaultInitialization: o,
      setDefaultInitialization: u,
      getDefaultOptions: c,
      setDefaultOptions: f
    });
  };
  yn.nonce = Ya;
  yn.trustedTypePolicy = qr;
  r.ClickScrollPlugin = Ia;
  r.OverlayScrollbars = yn;
  r.ScrollbarsHidingPlugin = za;
  r.SizeObserverPlugin = xa;
  return r;
}({});
//# sourceMappingURL=overlayscrollbars.browser.es5.js.map

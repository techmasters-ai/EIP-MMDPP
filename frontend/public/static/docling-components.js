/**
 * @license
 * Copyright 2017 Google LLC
 * SPDX-License-Identifier: BSD-3-Clause
 */
const v = (i) => (t, e) => {
  e !== void 0 ? e.addInitializer(() => {
    customElements.define(i, t);
  }) : customElements.define(i, t);
};
/**
 * @license
 * Copyright 2019 Google LLC
 * SPDX-License-Identifier: BSD-3-Clause
 */
const it = globalThis, Pt = it.ShadowRoot && (it.ShadyCSS === void 0 || it.ShadyCSS.nativeShadow) && "adoptedStyleSheets" in Document.prototype && "replace" in CSSStyleSheet.prototype, xt = Symbol(), Tt = /* @__PURE__ */ new WeakMap();
let Wt = class {
  constructor(t, e, s) {
    if (this._$cssResult$ = !0, s !== xt) throw Error("CSSResult is not constructable. Use `unsafeCSS` or `css` instead.");
    this.cssText = t, this.t = e;
  }
  get styleSheet() {
    let t = this.o;
    const e = this.t;
    if (Pt && t === void 0) {
      const s = e !== void 0 && e.length === 1;
      s && (t = Tt.get(e)), t === void 0 && ((this.o = t = new CSSStyleSheet()).replaceSync(this.cssText), s && Tt.set(e, t));
    }
    return t;
  }
  toString() {
    return this.cssText;
  }
};
const ie = (i) => new Wt(typeof i == "string" ? i : i + "", void 0, xt), O = (i, ...t) => {
  const e = i.length === 1 ? i[0] : t.reduce((s, r, o) => s + ((n) => {
    if (n._$cssResult$ === !0) return n.cssText;
    if (typeof n == "number") return n;
    throw Error("Value passed to 'css' function must be a 'css' function result: " + n + ". Use 'unsafeCSS' to pass non-literal values, but take care to ensure page security.");
  })(r) + i[o + 1], i[0]);
  return new Wt(e, i, xt);
}, oe = (i, t) => {
  if (Pt) i.adoptedStyleSheets = t.map((e) => e instanceof CSSStyleSheet ? e : e.styleSheet);
  else for (const e of t) {
    const s = document.createElement("style"), r = it.litNonce;
    r !== void 0 && s.setAttribute("nonce", r), s.textContent = e.cssText, i.appendChild(s);
  }
}, jt = Pt ? (i) => i : (i) => i instanceof CSSStyleSheet ? ((t) => {
  let e = "";
  for (const s of t.cssRules) e += s.cssText;
  return ie(e);
})(i) : i;
/**
 * @license
 * Copyright 2017 Google LLC
 * SPDX-License-Identifier: BSD-3-Clause
 */
const { is: ne, defineProperty: ae, getOwnPropertyDescriptor: he, getOwnPropertyNames: le, getOwnPropertySymbols: ce, getPrototypeOf: pe } = Object, j = globalThis, kt = j.trustedTypes, de = kt ? kt.emptyScript : "", ut = j.reactiveElementPolyfillSupport, J = (i, t) => i, ot = { toAttribute(i, t) {
  switch (t) {
    case Boolean:
      i = i ? de : null;
      break;
    case Object:
    case Array:
      i = i == null ? i : JSON.stringify(i);
  }
  return i;
}, fromAttribute(i, t) {
  let e = i;
  switch (t) {
    case Boolean:
      e = i !== null;
      break;
    case Number:
      e = i === null ? null : Number(i);
      break;
    case Object:
    case Array:
      try {
        e = JSON.parse(i);
      } catch {
        e = null;
      }
  }
  return e;
} }, ct = (i, t) => !ne(i, t), Ut = { attribute: !0, type: String, converter: ot, reflect: !1, useDefault: !1, hasChanged: ct };
Symbol.metadata ?? (Symbol.metadata = Symbol("metadata")), j.litPropertyMetadata ?? (j.litPropertyMetadata = /* @__PURE__ */ new WeakMap());
let L = class extends HTMLElement {
  static addInitializer(t) {
    this._$Ei(), (this.l ?? (this.l = [])).push(t);
  }
  static get observedAttributes() {
    return this.finalize(), this._$Eh && [...this._$Eh.keys()];
  }
  static createProperty(t, e = Ut) {
    if (e.state && (e.attribute = !1), this._$Ei(), this.prototype.hasOwnProperty(t) && ((e = Object.create(e)).wrapped = !0), this.elementProperties.set(t, e), !e.noAccessor) {
      const s = Symbol(), r = this.getPropertyDescriptor(t, s, e);
      r !== void 0 && ae(this.prototype, t, r);
    }
  }
  static getPropertyDescriptor(t, e, s) {
    const { get: r, set: o } = he(this.prototype, t) ?? { get() {
      return this[e];
    }, set(n) {
      this[e] = n;
    } };
    return { get: r, set(n) {
      const h = r == null ? void 0 : r.call(this);
      o == null || o.call(this, n), this.requestUpdate(t, h, s);
    }, configurable: !0, enumerable: !0 };
  }
  static getPropertyOptions(t) {
    return this.elementProperties.get(t) ?? Ut;
  }
  static _$Ei() {
    if (this.hasOwnProperty(J("elementProperties"))) return;
    const t = pe(this);
    t.finalize(), t.l !== void 0 && (this.l = [...t.l]), this.elementProperties = new Map(t.elementProperties);
  }
  static finalize() {
    if (this.hasOwnProperty(J("finalized"))) return;
    if (this.finalized = !0, this._$Ei(), this.hasOwnProperty(J("properties"))) {
      const e = this.properties, s = [...le(e), ...ce(e)];
      for (const r of s) this.createProperty(r, e[r]);
    }
    const t = this[Symbol.metadata];
    if (t !== null) {
      const e = litPropertyMetadata.get(t);
      if (e !== void 0) for (const [s, r] of e) this.elementProperties.set(s, r);
    }
    this._$Eh = /* @__PURE__ */ new Map();
    for (const [e, s] of this.elementProperties) {
      const r = this._$Eu(e, s);
      r !== void 0 && this._$Eh.set(r, e);
    }
    this.elementStyles = this.finalizeStyles(this.styles);
  }
  static finalizeStyles(t) {
    const e = [];
    if (Array.isArray(t)) {
      const s = new Set(t.flat(1 / 0).reverse());
      for (const r of s) e.unshift(jt(r));
    } else t !== void 0 && e.push(jt(t));
    return e;
  }
  static _$Eu(t, e) {
    const s = e.attribute;
    return s === !1 ? void 0 : typeof s == "string" ? s : typeof t == "string" ? t.toLowerCase() : void 0;
  }
  constructor() {
    super(), this._$Ep = void 0, this.isUpdatePending = !1, this.hasUpdated = !1, this._$Em = null, this._$Ev();
  }
  _$Ev() {
    var t;
    this._$ES = new Promise((e) => this.enableUpdating = e), this._$AL = /* @__PURE__ */ new Map(), this._$E_(), this.requestUpdate(), (t = this.constructor.l) == null || t.forEach((e) => e(this));
  }
  addController(t) {
    var e;
    (this._$EO ?? (this._$EO = /* @__PURE__ */ new Set())).add(t), this.renderRoot !== void 0 && this.isConnected && ((e = t.hostConnected) == null || e.call(t));
  }
  removeController(t) {
    var e;
    (e = this._$EO) == null || e.delete(t);
  }
  _$E_() {
    const t = /* @__PURE__ */ new Map(), e = this.constructor.elementProperties;
    for (const s of e.keys()) this.hasOwnProperty(s) && (t.set(s, this[s]), delete this[s]);
    t.size > 0 && (this._$Ep = t);
  }
  createRenderRoot() {
    const t = this.shadowRoot ?? this.attachShadow(this.constructor.shadowRootOptions);
    return oe(t, this.constructor.elementStyles), t;
  }
  connectedCallback() {
    var t;
    this.renderRoot ?? (this.renderRoot = this.createRenderRoot()), this.enableUpdating(!0), (t = this._$EO) == null || t.forEach((e) => {
      var s;
      return (s = e.hostConnected) == null ? void 0 : s.call(e);
    });
  }
  enableUpdating(t) {
  }
  disconnectedCallback() {
    var t;
    (t = this._$EO) == null || t.forEach((e) => {
      var s;
      return (s = e.hostDisconnected) == null ? void 0 : s.call(e);
    });
  }
  attributeChangedCallback(t, e, s) {
    this._$AK(t, s);
  }
  _$ET(t, e) {
    var o;
    const s = this.constructor.elementProperties.get(t), r = this.constructor._$Eu(t, s);
    if (r !== void 0 && s.reflect === !0) {
      const n = (((o = s.converter) == null ? void 0 : o.toAttribute) !== void 0 ? s.converter : ot).toAttribute(e, s.type);
      this._$Em = t, n == null ? this.removeAttribute(r) : this.setAttribute(r, n), this._$Em = null;
    }
  }
  _$AK(t, e) {
    var o, n;
    const s = this.constructor, r = s._$Eh.get(t);
    if (r !== void 0 && this._$Em !== r) {
      const h = s.getPropertyOptions(r), a = typeof h.converter == "function" ? { fromAttribute: h.converter } : ((o = h.converter) == null ? void 0 : o.fromAttribute) !== void 0 ? h.converter : ot;
      this._$Em = r, this[r] = a.fromAttribute(e, h.type) ?? ((n = this._$Ej) == null ? void 0 : n.get(r)) ?? null, this._$Em = null;
    }
  }
  requestUpdate(t, e, s) {
    var r;
    if (t !== void 0) {
      const o = this.constructor, n = this[t];
      if (s ?? (s = o.getPropertyOptions(t)), !((s.hasChanged ?? ct)(n, e) || s.useDefault && s.reflect && n === ((r = this._$Ej) == null ? void 0 : r.get(t)) && !this.hasAttribute(o._$Eu(t, s)))) return;
      this.C(t, e, s);
    }
    this.isUpdatePending === !1 && (this._$ES = this._$EP());
  }
  C(t, e, { useDefault: s, reflect: r, wrapped: o }, n) {
    s && !(this._$Ej ?? (this._$Ej = /* @__PURE__ */ new Map())).has(t) && (this._$Ej.set(t, n ?? e ?? this[t]), o !== !0 || n !== void 0) || (this._$AL.has(t) || (this.hasUpdated || s || (e = void 0), this._$AL.set(t, e)), r === !0 && this._$Em !== t && (this._$Eq ?? (this._$Eq = /* @__PURE__ */ new Set())).add(t));
  }
  async _$EP() {
    this.isUpdatePending = !0;
    try {
      await this._$ES;
    } catch (e) {
      Promise.reject(e);
    }
    const t = this.scheduleUpdate();
    return t != null && await t, !this.isUpdatePending;
  }
  scheduleUpdate() {
    return this.performUpdate();
  }
  performUpdate() {
    var s;
    if (!this.isUpdatePending) return;
    if (!this.hasUpdated) {
      if (this.renderRoot ?? (this.renderRoot = this.createRenderRoot()), this._$Ep) {
        for (const [o, n] of this._$Ep) this[o] = n;
        this._$Ep = void 0;
      }
      const r = this.constructor.elementProperties;
      if (r.size > 0) for (const [o, n] of r) {
        const { wrapped: h } = n, a = this[o];
        h !== !0 || this._$AL.has(o) || a === void 0 || this.C(o, void 0, n, a);
      }
    }
    let t = !1;
    const e = this._$AL;
    try {
      t = this.shouldUpdate(e), t ? (this.willUpdate(e), (s = this._$EO) == null || s.forEach((r) => {
        var o;
        return (o = r.hostUpdate) == null ? void 0 : o.call(r);
      }), this.update(e)) : this._$EM();
    } catch (r) {
      throw t = !1, this._$EM(), r;
    }
    t && this._$AE(e);
  }
  willUpdate(t) {
  }
  _$AE(t) {
    var e;
    (e = this._$EO) == null || e.forEach((s) => {
      var r;
      return (r = s.hostUpdated) == null ? void 0 : r.call(s);
    }), this.hasUpdated || (this.hasUpdated = !0, this.firstUpdated(t)), this.updated(t);
  }
  _$EM() {
    this._$AL = /* @__PURE__ */ new Map(), this.isUpdatePending = !1;
  }
  get updateComplete() {
    return this.getUpdateComplete();
  }
  getUpdateComplete() {
    return this._$ES;
  }
  shouldUpdate(t) {
    return !0;
  }
  update(t) {
    this._$Eq && (this._$Eq = this._$Eq.forEach((e) => this._$ET(e, this[e]))), this._$EM();
  }
  updated(t) {
  }
  firstUpdated(t) {
  }
};
L.elementStyles = [], L.shadowRootOptions = { mode: "open" }, L[J("elementProperties")] = /* @__PURE__ */ new Map(), L[J("finalized")] = /* @__PURE__ */ new Map(), ut == null || ut({ ReactiveElement: L }), (j.reactiveElementVersions ?? (j.reactiveElementVersions = [])).push("2.1.0");
/**
 * @license
 * Copyright 2017 Google LLC
 * SPDX-License-Identifier: BSD-3-Clause
 */
const ue = { attribute: !0, type: String, converter: ot, reflect: !1, hasChanged: ct }, me = (i = ue, t, e) => {
  const { kind: s, metadata: r } = e;
  let o = globalThis.litPropertyMetadata.get(r);
  if (o === void 0 && globalThis.litPropertyMetadata.set(r, o = /* @__PURE__ */ new Map()), s === "setter" && ((i = Object.create(i)).wrapped = !0), o.set(e.name, i), s === "accessor") {
    const { name: n } = e;
    return { set(h) {
      const a = t.get.call(this);
      t.set.call(this, h), this.requestUpdate(n, a, i);
    }, init(h) {
      return h !== void 0 && this.C(n, void 0, i, h), h;
    } };
  }
  if (s === "setter") {
    const { name: n } = e;
    return function(h) {
      const a = this[n];
      t.call(this, h), this.requestUpdate(n, a, i);
    };
  }
  throw Error("Unsupported decorator location: " + s);
};
function d(i) {
  return (t, e) => typeof e == "object" ? me(i, t, e) : ((s, r, o) => {
    const n = r.hasOwnProperty(o);
    return r.constructor.createProperty(o, s), n ? Object.getOwnPropertyDescriptor(r, o) : void 0;
  })(i, t, e);
}
const $t = [];
function Vt(i) {
  const t = v(i);
  return function(e, s) {
    return $t.push(
      e
    ), t(e, s);
  };
}
/**
 * @license
 * Copyright 2017 Google LLC
 * SPDX-License-Identifier: BSD-3-Clause
 */
const Z = globalThis, nt = Z.trustedTypes, Mt = nt ? nt.createPolicy("lit-html", { createHTML: (i) => i }) : void 0, Ft = "$lit$", T = `lit$${Math.random().toFixed(9).slice(2)}$`, Gt = "?" + T, fe = `<${Gt}>`, R = document, X = () => R.createComment(""), Y = (i) => i === null || typeof i != "object" && typeof i != "function", Ct = Array.isArray, ge = (i) => Ct(i) || typeof (i == null ? void 0 : i[Symbol.iterator]) == "function", mt = `[ 	
\f\r]`, K = /<(?:(!--|\/[^a-zA-Z])|(\/?[a-zA-Z][^>\s]*)|(\/?$))/g, Nt = /-->/g, Ht = />/g, M = RegExp(`>|${mt}(?:([^\\s"'>=/]+)(${mt}*=${mt}*(?:[^ 	
\f\r"'\`<>=]|("|')|))|$)`, "g"), Rt = /'/g, zt = /"/g, Kt = /^(?:script|style|textarea|title)$/i, Jt = (i) => (t, ...e) => ({ _$litType$: i, strings: t, values: e }), m = Jt(1), st = Jt(2), W = Symbol.for("lit-noChange"), f = Symbol.for("lit-nothing"), Bt = /* @__PURE__ */ new WeakMap(), N = R.createTreeWalker(R, 129);
function Zt(i, t) {
  if (!Ct(i) || !i.hasOwnProperty("raw")) throw Error("invalid template strings array");
  return Mt !== void 0 ? Mt.createHTML(t) : t;
}
const $e = (i, t) => {
  const e = i.length - 1, s = [];
  let r, o = t === 2 ? "<svg>" : t === 3 ? "<math>" : "", n = K;
  for (let h = 0; h < e; h++) {
    const a = i[h];
    let l, p, c = -1, u = 0;
    for (; u < a.length && (n.lastIndex = u, p = n.exec(a), p !== null); ) u = n.lastIndex, n === K ? p[1] === "!--" ? n = Nt : p[1] !== void 0 ? n = Ht : p[2] !== void 0 ? (Kt.test(p[2]) && (r = RegExp("</" + p[2], "g")), n = M) : p[3] !== void 0 && (n = M) : n === M ? p[0] === ">" ? (n = r ?? K, c = -1) : p[1] === void 0 ? c = -2 : (c = n.lastIndex - p[2].length, l = p[1], n = p[3] === void 0 ? M : p[3] === '"' ? zt : Rt) : n === zt || n === Rt ? n = M : n === Nt || n === Ht ? n = K : (n = M, r = void 0);
    const g = n === M && i[h + 1].startsWith("/>") ? " " : "";
    o += n === K ? a + fe : c >= 0 ? (s.push(l), a.slice(0, c) + Ft + a.slice(c) + T + g) : a + T + (c === -2 ? h : g);
  }
  return [Zt(i, o + (i[e] || "<?>") + (t === 2 ? "</svg>" : t === 3 ? "</math>" : "")), s];
};
class Q {
  constructor({ strings: t, _$litType$: e }, s) {
    let r;
    this.parts = [];
    let o = 0, n = 0;
    const h = t.length - 1, a = this.parts, [l, p] = $e(t, e);
    if (this.el = Q.createElement(l, s), N.currentNode = this.el.content, e === 2 || e === 3) {
      const c = this.el.content.firstChild;
      c.replaceWith(...c.childNodes);
    }
    for (; (r = N.nextNode()) !== null && a.length < h; ) {
      if (r.nodeType === 1) {
        if (r.hasAttributes()) for (const c of r.getAttributeNames()) if (c.endsWith(Ft)) {
          const u = p[n++], g = r.getAttribute(c).split(T), _ = /([.?@])?(.*)/.exec(u);
          a.push({ type: 1, index: o, name: _[2], strings: g, ctor: _[1] === "." ? ye : _[1] === "?" ? ve : _[1] === "@" ? be : pt }), r.removeAttribute(c);
        } else c.startsWith(T) && (a.push({ type: 6, index: o }), r.removeAttribute(c));
        if (Kt.test(r.tagName)) {
          const c = r.textContent.split(T), u = c.length - 1;
          if (u > 0) {
            r.textContent = nt ? nt.emptyScript : "";
            for (let g = 0; g < u; g++) r.append(c[g], X()), N.nextNode(), a.push({ type: 2, index: ++o });
            r.append(c[u], X());
          }
        }
      } else if (r.nodeType === 8) if (r.data === Gt) a.push({ type: 2, index: o });
      else {
        let c = -1;
        for (; (c = r.data.indexOf(T, c + 1)) !== -1; ) a.push({ type: 7, index: o }), c += T.length - 1;
      }
      o++;
    }
  }
  static createElement(t, e) {
    const s = R.createElement("template");
    return s.innerHTML = t, s;
  }
}
function V(i, t, e = i, s) {
  var n, h;
  if (t === W) return t;
  let r = s !== void 0 ? (n = e._$Co) == null ? void 0 : n[s] : e._$Cl;
  const o = Y(t) ? void 0 : t._$litDirective$;
  return (r == null ? void 0 : r.constructor) !== o && ((h = r == null ? void 0 : r._$AO) == null || h.call(r, !1), o === void 0 ? r = void 0 : (r = new o(i), r._$AT(i, e, s)), s !== void 0 ? (e._$Co ?? (e._$Co = []))[s] = r : e._$Cl = r), r !== void 0 && (t = V(i, r._$AS(i, t.values), r, s)), t;
}
class _e {
  constructor(t, e) {
    this._$AV = [], this._$AN = void 0, this._$AD = t, this._$AM = e;
  }
  get parentNode() {
    return this._$AM.parentNode;
  }
  get _$AU() {
    return this._$AM._$AU;
  }
  u(t) {
    const { el: { content: e }, parts: s } = this._$AD, r = ((t == null ? void 0 : t.creationScope) ?? R).importNode(e, !0);
    N.currentNode = r;
    let o = N.nextNode(), n = 0, h = 0, a = s[0];
    for (; a !== void 0; ) {
      if (n === a.index) {
        let l;
        a.type === 2 ? l = new et(o, o.nextSibling, this, t) : a.type === 1 ? l = new a.ctor(o, a.name, a.strings, this, t) : a.type === 6 && (l = new we(o, this, t)), this._$AV.push(l), a = s[++h];
      }
      n !== (a == null ? void 0 : a.index) && (o = N.nextNode(), n++);
    }
    return N.currentNode = R, r;
  }
  p(t) {
    let e = 0;
    for (const s of this._$AV) s !== void 0 && (s.strings !== void 0 ? (s._$AI(t, s, e), e += s.strings.length - 2) : s._$AI(t[e])), e++;
  }
}
class et {
  get _$AU() {
    var t;
    return ((t = this._$AM) == null ? void 0 : t._$AU) ?? this._$Cv;
  }
  constructor(t, e, s, r) {
    this.type = 2, this._$AH = f, this._$AN = void 0, this._$AA = t, this._$AB = e, this._$AM = s, this.options = r, this._$Cv = (r == null ? void 0 : r.isConnected) ?? !0;
  }
  get parentNode() {
    let t = this._$AA.parentNode;
    const e = this._$AM;
    return e !== void 0 && (t == null ? void 0 : t.nodeType) === 11 && (t = e.parentNode), t;
  }
  get startNode() {
    return this._$AA;
  }
  get endNode() {
    return this._$AB;
  }
  _$AI(t, e = this) {
    t = V(this, t, e), Y(t) ? t === f || t == null || t === "" ? (this._$AH !== f && this._$AR(), this._$AH = f) : t !== this._$AH && t !== W && this._(t) : t._$litType$ !== void 0 ? this.$(t) : t.nodeType !== void 0 ? this.T(t) : ge(t) ? this.k(t) : this._(t);
  }
  O(t) {
    return this._$AA.parentNode.insertBefore(t, this._$AB);
  }
  T(t) {
    this._$AH !== t && (this._$AR(), this._$AH = this.O(t));
  }
  _(t) {
    this._$AH !== f && Y(this._$AH) ? this._$AA.nextSibling.data = t : this.T(R.createTextNode(t)), this._$AH = t;
  }
  $(t) {
    var o;
    const { values: e, _$litType$: s } = t, r = typeof s == "number" ? this._$AC(t) : (s.el === void 0 && (s.el = Q.createElement(Zt(s.h, s.h[0]), this.options)), s);
    if (((o = this._$AH) == null ? void 0 : o._$AD) === r) this._$AH.p(e);
    else {
      const n = new _e(r, this), h = n.u(this.options);
      n.p(e), this.T(h), this._$AH = n;
    }
  }
  _$AC(t) {
    let e = Bt.get(t.strings);
    return e === void 0 && Bt.set(t.strings, e = new Q(t)), e;
  }
  k(t) {
    Ct(this._$AH) || (this._$AH = [], this._$AR());
    const e = this._$AH;
    let s, r = 0;
    for (const o of t) r === e.length ? e.push(s = new et(this.O(X()), this.O(X()), this, this.options)) : s = e[r], s._$AI(o), r++;
    r < e.length && (this._$AR(s && s._$AB.nextSibling, r), e.length = r);
  }
  _$AR(t = this._$AA.nextSibling, e) {
    var s;
    for ((s = this._$AP) == null ? void 0 : s.call(this, !1, !0, e); t && t !== this._$AB; ) {
      const r = t.nextSibling;
      t.remove(), t = r;
    }
  }
  setConnected(t) {
    var e;
    this._$AM === void 0 && (this._$Cv = t, (e = this._$AP) == null || e.call(this, t));
  }
}
class pt {
  get tagName() {
    return this.element.tagName;
  }
  get _$AU() {
    return this._$AM._$AU;
  }
  constructor(t, e, s, r, o) {
    this.type = 1, this._$AH = f, this._$AN = void 0, this.element = t, this.name = e, this._$AM = r, this.options = o, s.length > 2 || s[0] !== "" || s[1] !== "" ? (this._$AH = Array(s.length - 1).fill(new String()), this.strings = s) : this._$AH = f;
  }
  _$AI(t, e = this, s, r) {
    const o = this.strings;
    let n = !1;
    if (o === void 0) t = V(this, t, e, 0), n = !Y(t) || t !== this._$AH && t !== W, n && (this._$AH = t);
    else {
      const h = t;
      let a, l;
      for (t = o[0], a = 0; a < o.length - 1; a++) l = V(this, h[s + a], e, a), l === W && (l = this._$AH[a]), n || (n = !Y(l) || l !== this._$AH[a]), l === f ? t = f : t !== f && (t += (l ?? "") + o[a + 1]), this._$AH[a] = l;
    }
    n && !r && this.j(t);
  }
  j(t) {
    t === f ? this.element.removeAttribute(this.name) : this.element.setAttribute(this.name, t ?? "");
  }
}
class ye extends pt {
  constructor() {
    super(...arguments), this.type = 3;
  }
  j(t) {
    this.element[this.name] = t === f ? void 0 : t;
  }
}
class ve extends pt {
  constructor() {
    super(...arguments), this.type = 4;
  }
  j(t) {
    this.element.toggleAttribute(this.name, !!t && t !== f);
  }
}
class be extends pt {
  constructor(t, e, s, r, o) {
    super(t, e, s, r, o), this.type = 5;
  }
  _$AI(t, e = this) {
    if ((t = V(this, t, e, 0) ?? f) === W) return;
    const s = this._$AH, r = t === f && s !== f || t.capture !== s.capture || t.once !== s.once || t.passive !== s.passive, o = t !== f && (s === f || r);
    r && this.element.removeEventListener(this.name, this, s), o && this.element.addEventListener(this.name, this, t), this._$AH = t;
  }
  handleEvent(t) {
    var e;
    typeof this._$AH == "function" ? this._$AH.call(((e = this.options) == null ? void 0 : e.host) ?? this.element, t) : this._$AH.handleEvent(t);
  }
}
class we {
  constructor(t, e, s) {
    this.element = t, this.type = 6, this._$AN = void 0, this._$AM = e, this.options = s;
  }
  get _$AU() {
    return this._$AM._$AU;
  }
  _$AI(t) {
    V(this, t);
  }
}
const ft = Z.litHtmlPolyfillSupport;
ft == null || ft(Q, et), (Z.litHtmlVersions ?? (Z.litHtmlVersions = [])).push("3.3.0");
const Ae = (i, t, e) => {
  const s = (e == null ? void 0 : e.renderBefore) ?? t;
  let r = s._$litPart$;
  if (r === void 0) {
    const o = (e == null ? void 0 : e.renderBefore) ?? null;
    s._$litPart$ = r = new et(t.insertBefore(X(), o), o, void 0, e ?? {});
  }
  return r._$AI(i), r;
};
/**
 * @license
 * Copyright 2017 Google LLC
 * SPDX-License-Identifier: BSD-3-Clause
 */
const H = globalThis;
let A = class extends L {
  constructor() {
    super(...arguments), this.renderOptions = { host: this }, this._$Do = void 0;
  }
  createRenderRoot() {
    var e;
    const t = super.createRenderRoot();
    return (e = this.renderOptions).renderBefore ?? (e.renderBefore = t.firstChild), t;
  }
  update(t) {
    const e = this.render();
    this.hasUpdated || (this.renderOptions.isConnected = this.isConnected), super.update(t), this._$Do = Ae(e, this.renderRoot, this.renderOptions);
  }
  connectedCallback() {
    var t;
    super.connectedCallback(), (t = this._$Do) == null || t.setConnected(!0);
  }
  disconnectedCallback() {
    var t;
    super.disconnectedCallback(), (t = this._$Do) == null || t.setConnected(!1);
  }
  render() {
    return W;
  }
};
var qt;
A._$litElement$ = !0, A.finalized = !0, (qt = H.litElementHydrateSupport) == null || qt.call(H, { LitElement: A });
const gt = H.litElementPolyfillSupport;
gt == null || gt({ LitElement: A });
(H.litElementVersions ?? (H.litElementVersions = [])).push("4.2.0");
var Pe = Object.defineProperty, dt = (i, t, e, s) => {
  for (var r = void 0, o = i.length - 1, n; o >= 0; o--)
    (n = i[o]) && (r = n(t, e, r) || r);
  return r && Pe(t, e, r), r;
};
class z extends A {
  render() {
    if (this.annotation && this.item && this.page && this.canDrawAnnotation(this.annotation) && this.canDrawItem(this.item))
      return this.renderAnnotation(
        this.annotation,
        this.item,
        this.page,
        this.prov
      );
  }
}
dt([
  d({ attribute: !1 })
], z.prototype, "annotation");
dt([
  d({ attribute: !1 })
], z.prototype, "item");
dt([
  d({ attribute: !1 })
], z.prototype, "page");
dt([
  d({ attribute: !1 })
], z.prototype, "prov");
function B(...i) {
  return function(t) {
    return b.DocItem(t) && i.includes(t.label);
  };
}
var I = {
  CodeItem: B("code"),
  ListItem: B("list_item"),
  PictureItem: B("chart", "picture"),
  SectionHeaderItem: B("section_header"),
  TableItem: B("document_index", "table"),
  TextItem: B(
    "caption",
    "checkbox_selected",
    "checkbox_unselected",
    "footnote",
    "page_footer",
    "page_header",
    "paragraph",
    "reference",
    "text"
  )
};
function S(...i) {
  return function(t) {
    return i.includes(t.kind);
  };
}
var Ot = {
  PictureBarChart: S("bar_chart_data"),
  PictureClassification: S("classification"),
  PictureDescription: S("description"),
  PictureMisc: S("misc"),
  PictureMolecule: S("molecule_data"),
  PictureLineChart: S("line_chart_data"),
  PicturePieChart: S("pie_chart_data"),
  PictureScatterChart: S("scatter_chart_data"),
  PictureStackedBarChart: S(
    "stacked_bar_chart_data"
  )
}, b = {
  DocItem(i) {
    return b.NodeItem(i) && !b.GroupItem(i);
  },
  Document(i) {
    return "schema_name" in i && i.schema_name === "DoclingDocument";
  },
  GroupItem(i) {
    return b.NodeItem(i) && (i.self_ref.startsWith("#/groups/") || i.self_ref === "#/body");
  },
  NodeItem(i) {
    return "self_ref" in i;
  },
  ...Ot,
  ...I
};
function* xe(i, t = {}) {
  i && (yield* e(t.root ?? i.body));
  function* e(s, r = 0) {
    var o;
    if ((!b.GroupItem(s) || t.withGroups) && (b.DocItem(s) ? (t.pageNo === void 0 || (o = s.prov) != null && o.some((n) => n.page_no === t.pageNo)) && (yield [s, r]) : yield [s, r]), !(b.PictureItem(s) && !t.traversePictures))
      for (const n of s.children ?? []) {
        const h = Ce(i, n);
        b.NodeItem(h) && (yield* e(h, r + 1));
      }
  }
}
function Ce(i, t) {
  return t.$ref.split("/").slice(1).reduce(
    (s, r) => s[r],
    i
  );
}
var Oe = Object.defineProperty, Ee = Object.getOwnPropertyDescriptor, Xt = (i, t, e, s) => {
  for (var r = s > 1 ? void 0 : s ? Ee(t, e) : t, o = i.length - 1, n; o >= 0; o--)
    (n = i[o]) && (r = (s ? n(t, e, r) : n(r)) || r);
  return s && r && Oe(t, e, r), r;
};
let at = class extends z {
  constructor() {
    super(...arguments), this.precision = 2;
  }
  renderAnnotation(i, t, e, s) {
    const r = i.predicted_classes ?? [], o = Math.pow(10, -1 * this.precision), n = r.filter((l) => o < l.confidence), h = r.filter((l) => l.confidence < o), a = (l) => l.toLocaleString(void 0, {
      maximumFractionDigits: this.precision
    });
    return m`<table>
      <thead>
        <tr>
          <th>Class</th>
          <th>Confidence</th>
        </tr>
      </thead>
      <tbody>
        ${n.map(
      ({ class_name: l, confidence: p }) => m`<tr>
              <td>${l}</td>
              <td>${a(p)}</td>
            </tr>`
    )}
        ${h.length === 0 ? f : m`<tr
              class="more"
              title=${h.map((l) => `${l.class_name}			${l.confidence}`).join(`
`)}
            >
              <td>${h.length} more</td>
              <td>< ${a(o)}</td>
            </tr>`}
      </tbody>
    </table>`;
  }
  canDrawAnnotation(i) {
    return Ot.PictureClassification(i);
  }
  canDrawItem(i) {
    return I.PictureItem(i);
  }
};
at.styles = O`
    table {
      margin: 0.5rem;
      font-size: inherit;
      line-height: 1.25;
    }

    td,
    th {
      padding: 0 1rem 0 0;
    }

    th {
      font-weight: bold;
      text-align: left;
    }

    tr.more td {
      padding-top: 0.25rem;
      color: gray;
    }
  `;
Xt([
  d()
], at.prototype, "precision", 2);
at = Xt([
  Vt("docling-picture-classification")
], at);
var Se = Object.getOwnPropertyDescriptor, Ie = (i, t, e, s) => {
  for (var r = s > 1 ? void 0 : s ? Se(t, e) : t, o = i.length - 1, n; o >= 0; o--)
    (n = i[o]) && (r = n(r) || r);
  return r;
};
let _t = class extends z {
  constructor() {
    super();
    this.scrollSpeed = 15; // px per second
    this.maxHeight = 250;  // visible viewport height (px)

    this._container = null;
    this._content = null;
    this._isRunning = false;

    this._onSelfEnter = () => this._pause(true);
    this._onSelfLeave = () => this._pause(false);
  }

  connectedCallback() {
    super.connectedCallback();
    // When the tooltip exists, the user is hovering the image—start scroll
    // after first render measures.
    this.updateComplete?.then(() => this._setupAndMaybeStart());
    this.addEventListener('mouseenter', this._onSelfEnter);
    this.addEventListener('mouseleave', this._onSelfLeave);
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    this.removeEventListener('mouseenter', this._onSelfEnter);
    this.removeEventListener('mouseleave', this._onSelfLeave);
    this._stop();
  }

  firstUpdated() {
    this._container = this.renderRoot.querySelector('.scroll-viewport');
    this._content = this.renderRoot.querySelector('.scroll-content');
    if (this._container) this._container.style.maxHeight = `${this.maxHeight}px`;
  }

  _setupAndMaybeStart() {
    this._container = this.renderRoot.querySelector('.scroll-viewport');
    this._content = this.renderRoot.querySelector('.scroll-content');
    if (!this._container || !this._content) return;

    // Only scroll if there’s overflow
    const overflow = this._content.scrollHeight - this._container.clientHeight;
    if (overflow > 4) {
      const durationSec = overflow / this.scrollSpeed;
      this._content.style.setProperty('--scroll-distance', `${overflow}px`);
      this._content.style.setProperty('--scroll-duration', `${durationSec}s`);
      this._start();
    } else {
      this._stop();
    }
  }

  _start() {
    if (!this._content || this._isRunning) return;
    this._content.classList.add('running');
    this._isRunning = true;
  }

  _stop() {
    if (!this._content) return;
    this._content.classList.remove('running', 'paused');
    this._content.style.removeProperty('--scroll-distance');
    this._content.style.removeProperty('--scroll-duration');
    this._content.style.transform = 'translateY(0)';
    this._isRunning = false;
  }

  _pause(paused) {
    if (!this._content || !this._isRunning) return;
    this._content.classList.toggle('paused', paused);
  }

  renderAnnotation(i /* annotation */) {
    return m`
      <div class="container">
        <p class="label"><span>AI Image Analysis:</span></p>
        <div class="scroll-viewport" part="scroll-viewport">
          <div class="scroll-content" part="scroll-content">${i.text}</div>
        </div>
      </div>
    `;
  }

  canDrawAnnotation(i) { return Ot.PictureDescription(i); }
  canDrawItem(i) { return I.PictureItem(i); }
};

_t.styles = O`
  .container {
    margin: 0.5rem;
    font-size: 0.9rem;
    line-height: 1.25;
    color: inherit;
  }
  .label { margin: 0 0 0.25rem 0; }
  .label > span { padding-right: 0.5rem; font-weight: bold; }

  .scroll-viewport {
    position: relative;
    overflow: hidden;
    max-width: 32rem; /* optional: avoid super-wide tooltips */
  }
  .scroll-content {
    white-space: pre-line;
    will-change: transform;
    transform: translateY(0);
  }

  /* animate only while mounted and not paused */
  .scroll-content.running:not(.paused) {
    animation-name: docling-auto-scroll-y;
    animation-duration: var(--scroll-duration, 10s);
    animation-timing-function: linear;
    animation-iteration-count: infinite;
    animation-direction: alternate; /* down then up for readability */
  }
  .scroll-content.paused { animation-play-state: paused; }

  @keyframes docling-auto-scroll-y {
    from { transform: translateY(0); }
    to   { transform: translateY(calc(-1 * var(--scroll-distance, 0px))); }
  }
`;

_t = Ie([ Vt("docling-picture-description") ], _t);
const yt = [];
function Yt(i) {
  const t = v(i);
  return function(e, s) {
    return yt.push(e), t(e, s);
  };
}
var De = Object.defineProperty, Et = (i, t, e, s) => {
  for (var r = void 0, o = i.length - 1, n; o >= 0; o--)
    (n = i[o]) && (r = n(t, e, r) || r);
  return r && De(t, e, r), r;
};
class D extends A {
  render() {
    if (this.item && this.page && this.canDrawItem(this.item))
      return this.renderItem(this.item, this.page, this.prov);
  }
}
Et([
  d({ attribute: !1 })
], D.prototype, "item");
Et([
  d({ attribute: !1 })
], D.prototype, "page");
Et([
  d({ attribute: !1 })
], D.prototype, "prov");
var Te = Object.getOwnPropertyDescriptor, je = (i, t, e, s) => {
  for (var r = s > 1 ? void 0 : s ? Te(t, e) : t, o = i.length - 1, n; o >= 0; o--)
    (n = i[o]) && (r = n(r) || r);
  return r;
};
let ht = class extends D {
  renderItem(i, t) {
    var r;
    const { image: e } = t, s = (r = i.prov) == null ? void 0 : r.find((o) => {
      var n;
      return o.page_no === ((n = this.page) == null ? void 0 : n.page_no);
    });
    if (e && s) {
      const { width: o = 1, height: n = 1 } = this.page.size, { l: h, r: a, t: l, b: p } = q(s.bbox, t);
      return m`
        <svg
          width=${(a - h) * ((e.size.width ?? 1) / o)}
          viewBox="${h} ${l} ${a - h} ${p - l}"
        >
          <image href=${e.uri} width=${o} height=${n} />
        </svg>
      `;
    } else
      return m`<span>Invalid provenance.</span>`;
  }
  canDrawItem(i) {
    return b.DocItem(i);
  }
};
ht.styles = O`
    svg {
      max-width: 100%;
    }
  `;
ht = je([
  v("docling-item-provenance")
], ht);
var ke = Object.getOwnPropertyDescriptor, Ue = (i, t, e, s) => {
  for (var r = s > 1 ? void 0 : s ? ke(t, e) : t, o = i.length - 1, n; o >= 0; o--)
    (n = i[o]) && (r = n(r) || r);
  return r;
};
let vt = class extends D {
  renderItem(i) {
    const t = /* @__PURE__ */ new Set();
    return m`
      <div class="container">
        <table>
          <tbody>
            ${i.data.grid.map(
      (s) => m`<tr>
                  ${s.map((r) => {
        if (!e(r))
          return m`<td
                        class=${r.column_header || r.row_header ? "header" : ""}
                        colspan=${r.col_span ?? 1}
                        rowspan=${r.row_span ?? 1}
                      >
                        ${r.text}
                      </td>`;
      })}
                </tr>`
    )}
          </tbody>
        </table>
      </div>
    `;
    function e(s) {
      const r = t.has(
        [s.start_col_offset_idx, s.start_row_offset_idx].join()
      );
      if (!r)
        for (let o = s.start_col_offset_idx; o < s.end_col_offset_idx; o++)
          for (let n = s.start_row_offset_idx; n < s.end_row_offset_idx; n++)
            t.add([o, n].join());
      return r;
    }
  }
  canDrawItem(i) {
    return I.TableItem(i);
  }
};
vt.styles = O`
    .container {
      position: relative;
      min-width: 0;
      max-width: 100%;
      min-height: 0;
      overflow: auto;
    }

    table {
      border-collapse: collapse;
      font-size: 75%;
      line-height: 1.25;
    }

    td {
      padding: 0.125rem 0.25rem;
      background-color: var(--cds-layer);
      border: 1px solid rgb(220, 220, 220);

      color: black;
      text-decoration: none;
      word-break: normal;
      text-align: left;
    }

    td.header {
      font-weight: bold;
    }

    td:hover {
      filter: brightness(95%);
    }

    td:target {
      outline: 3px solid blue;
    }
  `;
vt = Ue([
  Yt("docling-item-table")
], vt);
var Me = Object.getOwnPropertyDescriptor, Ne = (i, t, e, s) => {
  for (var r = s > 1 ? void 0 : s ? Me(t, e) : t, o = i.length - 1, n; o >= 0; o--)
    (n = i[o]) && (r = n(r) || r);
  return r;
};
let Lt = class extends D {
  constructor() {
    super(...arguments), this.renderItem = Function(
      "item",
      "page",
      `"use strict"; return this.html\`${this.innerHTML}\`;`
    ).bind({ html: m });
  }
  canDrawItem(i) {
    return b.DocItem(i);
  }
};
Lt = Ne([
  v("docling-item-template")
], Lt);
var He = Object.getOwnPropertyDescriptor, Re = (i, t, e, s) => {
  for (var r = s > 1 ? void 0 : s ? He(t, e) : t, o = i.length - 1, n; o >= 0; o--)
    (n = i[o]) && (r = n(r) || r);
  return r;
};
let bt = class extends D {
  renderItem(i, t, e) {
    const s = I.SectionHeaderItem(i) ? "header" : "";
    if (e) {
      const { l: r, r: o, t: n, b: h } = q(e.bbox, t), a = Math.sqrt((o - r) * (h - n) / i.text.length), l = Math.min(Math.floor(1.25 * a), (h - n) / 1.25), [p, c] = e.charspan ?? [
        0,
        i.text.length
      ], u = i.text.substring(p, c);
      return m`<p
        class=${s}
        style="font-size: ${l}px; line-height: ${1.25 * l}px"
      >
        ${u}
      </p>`;
    } else
      return m`<p class=${s}>${i.text}</p>`;
  }
  canDrawItem(i) {
    return I.TextItem(i) || I.SectionHeaderItem(i) || I.ListItem(i);
  }
};
bt.styles = O`
    p {
      margin: 0;
      overflow-wrap: anywhere;
      font-size: 1rem;
      line-height: 1.25rem;
    }

    .header {
      font-weight: bold;
    }
  `;
bt = Re([
  Yt("docling-item-text")
], bt);
var ze = Object.getOwnPropertyDescriptor, St = (i, t, e, s) => {
  for (var r = s > 1 ? void 0 : s ? ze(t, e) : t, o = i.length - 1, n; o >= 0; o--)
    (n = i[o]) && (r = n(r) || r);
  return r;
};
let tt = class extends D {
  get itemChildren() {
    return Array.from(this.childNodes ?? []).filter(
      (i) => i instanceof D
    );
  }
  get annotationChildren() {
    return Array.from(this.childNodes).filter(
      (i) => i instanceof z
    );
  }
  get isCustomized() {
    return this.itemChildren.length > 0 || this.annotationChildren.length > 0;
  }
  renderItem(i, t, e) {
    const s = [], r = this.isCustomized;
    r ? this.itemChildren.filter((n) => n.canDrawItem(i)).forEach((n) => s.push(n)) : yt.filter((n) => n.prototype.canDrawItem(i)).forEach((n) => s.push(new n()));
    const o = i.annotations ?? [];
    for (const n of o) {
      const h = [];
      r ? this.annotationChildren.filter((a) => a.canDrawItem(a) && a.canDrawAnnotation(n)).forEach(
        (a) => h.push(a.cloneNode(!0))
      ) : $t.filter(
        (a) => a.prototype.canDrawItem(i) && a.prototype.canDrawAnnotation(n)
      ).forEach((a) => h.push(new a()));
      for (const a of h)
        a.annotation = n;
      s.push(...h);
    }
    for (const n of s)
      n.item = i, n.page = t, n.prov = e;
    return m`${s}`;
  }
  canDrawItem(i) {
    return b.DocItem(i) && this.isCustomized ? this.itemChildren.some((t) => t.canDrawItem(i)) || this.annotationChildren.some(
      (t) => {
        var e;
        return t.canDrawItem(i) && ((e = i.annotations) == null ? void 0 : e.some((s) => t.canDrawAnnotation(s)));
      }
    ) : yt.some((t) => t.prototype.canDrawItem(i)) || $t.some(
      (t) => {
        var e;
        return t.prototype.canDrawItem(i) && ((e = i.annotations) == null ? void 0 : e.some(
          (s) => t.prototype.canDrawAnnotation(s)
        ));
      }
    );
  }
};
tt = St([
  v("docling-view")
], tt);
let wt = class extends tt {
  constructor() {
    super(...arguments), this.type = "overlay";
  }
};
wt = St([
  v("docling-overlay")
], wt);
let At = class extends tt {
  constructor() {
    super(...arguments), this.type = "tooltip";
  }
};
At = St([
  v("docling-tooltip")
], At);
function q(i, t) {
  const { height: e = 1 } = t.size;
  return i.coord_origin === "TOPLEFT" ? i : { ...i, t: e - i.t, b: e - i.b };
}
var Be = Object.defineProperty, Le = Object.getOwnPropertyDescriptor, P = (i, t, e, s) => {
  for (var r = s > 1 ? void 0 : s ? Le(t, e) : t, o = i.length - 1, n; o >= 0; o--)
    (n = i[o]) && (r = (s ? n(t, e, r) : n(r)) || r);
  return s && r && Be(t, e, r), r;
};
let y = class extends A {
  constructor() {
    super(...arguments), this.pagenumbers = !1;
  }
  render() {
    var i, t, e;
    if ((i = this.page) != null && i.image) {
      const { image: s, size: r, page_no: o } = this.page, { width: n = 1, height: h = 1 } = r, a = this.onClickItem && this.page ? (l, p) => {
        var c;
        return (c = this.onClickItem) == null ? void 0 : c.call(this, l, this.page, p);
      } : void 0;
      return m`
        <div class="page" @onclick=${(l) => a == null ? void 0 : a(l)}>
          <svg
            class="base"
            width=${s.size.width}
            viewBox="0 0 ${n} ${h}"
          >
            ${this.backdrop ? st`
                <image
                  class="backdrop"
                  href=${s.uri}
                  width=${n}
                  height=${h}
                />

                <clippath id="clip-page-${o}">
                  ${(t = this.items) == null ? void 0 : t.map((l) => {
        var c;
        const p = (c = l.prov) == null ? void 0 : c.find((u) => u.page_no === o);
        if (p) {
          const { l: u, r: g, t: _, b: $ } = q(p.bbox, this.page);
          return st`<rect
                        x=${u}
                        y=${_}
                        width=${g - u}
                        height=${$ - _}
                      />`;
        }
      })}
                </clippath>
              ` : f}

            <image
              id="image"
              href=${s.uri}
              clip-path="url(#clip-page-${o})"
              width=${n}
              height=${h}
            />

            ${(e = this.items) == null ? void 0 : e.flatMap(
        (l) => (l.prov ?? []).filter((p) => p.page_no === o).map((p) => {
          var $;
          const { l: c, r: u, t: g, b: _ } = q(p.bbox, this.page);
          return st`
                  ${this.layOver(l, p)}

                  <rect
                  part=${"item" + (this.itemPart ? " " + this.itemPart(this.page, l) : "")}
                  style=${($ = this.itemStyle) == null ? void 0 : $.call(this, this.page, l)}
                  x=${c}
                  y=${g}
                  width=${u - c}
                  height=${_ - g}
                  vector-effect="non-scaling-stroke"
                  @click=${(rt) => {
            rt.stopPropagation(), a == null || a(rt, l);
          }}
                  @mouseenter=${(rt) => {
            const G = rt.currentTarget.getBoundingClientRect(), Dt = [
              G.top * window.innerWidth,
              (window.innerWidth - G.right) * window.innerHeight,
              (window.innerHeight - G.bottom) * window.innerWidth,
              G.left * window.innerHeight
            ], ee = Math.max(...Dt), re = Dt.findIndex((se) => se === ee);
            this.attachTooltip(l, p, G, re);
          }}
                  @mouseleave=${() => this.removeTooltip()}
                  />
                `;
        })
      )}
          </svg>

          ${this.renderTrace()}
          ${this.pagenumbers ? m`<header
                  part="page-number-top"
                  class="page-number-top"
                  title="Page ${o}"
                >
                  ${o}
                </header>

                <header
                  part="page-number-bottom"
                  class="page-number-bottom"
                  title="Page ${o}"
                >
                  ${o}
                </header>` : f}
        </div>
      `;
    } else
      return m`Invalid page image.`;
  }
  layOver(i, t) {
    var e;
    if ((e = this.overlay) != null && e.canDrawItem(i)) {
      const s = this.overlay.cloneNode(!0);
      s.item = i, s.page = this.page, s.prov = t;
      const { l: r, r: o, t: n, b: h } = q(t.bbox, this.page), a = I.PictureItem(i) ? "softOverlay" : "hardOverlay";
      return st`<foreignObject part="overlay" class=${a} x=${r} y=${n} width=${o - r} height=${h - n}>${s}</foreignObject>`;
    }
  }
  attachTooltip(i, t, e, s) {
    var r;
    if ((r = this.tooltip) != null && r.canDrawItem(i)) {
      const o = this.tooltip.cloneNode(!0);
      o.id = "tooltip", o.setAttribute("part", "tooltip"), o.className = "tooltip", o.setAttribute(
        "style",
        `
        ${s === 1 ? `left: ${e.right}px` : s === 3 ? `right: ${window.outerWidth - e.left}px` : `left: calc(${e.left}px - 2rem)`};
        ${s === 0 ? `bottom: ${window.innerHeight - e.top}px` : s === 2 ? `top: ${e.bottom}px` : `top: calc(${e.top}px - 2rem)`};
        max-width: ${2 * Math.max(t.bbox.r - t.bbox.l, 200)}px;
        `
      ), o.item = i, o.page = this.page, this.renderRoot.appendChild(o);
    }
  }
  removeTooltip() {
    var i;
    (i = this.renderRoot.querySelector("#tooltip")) == null || i.remove();
  }
  renderTrace() {
    if (this.trace) {
      const i = this.trace.cloneNode(!0);
      return i.page = this.page, i.items = this.items, i;
    }
  }
};
y.styles = O`
    .page {
      position: relative;
      width: fit-content;
      color: black;
    }

    .base {
      max-width: 100%;
    }

    svg:not(.base) {
      position: absolute;
      inset: 0;
    }

    .backdrop {
      opacity: 0.3;
    }

    .page-number-top,
    .page-number-bottom {
      position: absolute;
      left: 0;
      width: fit-content;
      padding: 0 0.25rem;

      font-size: 0.75rem;
      line-height: 1rem;
      color: rgb(120, 120, 120);
      mix-blend-mode: difference;
    }

    .page-number-top {
      top: 0;
    }

    .page-number-bottom {
      bottom: 0;
    }

    rect {
      fill: blue;
      fill-opacity: 0.0001; /* To activate hover. */
      stroke: grey;
      stroke-width: 1px;
      stroke-dasharray: 1;
      cursor: pointer;
    }

    rect:hover {
      fill-opacity: 0.1;
      stroke: blue;
    }

    rect:target {
      stroke: blue;
      stroke-width: 3px;
      stroke-dasharray: none;
    }

    .hardOverlay,
    .softOverlay {
      font-size: 62.5%;
      background-color: white;
    }

    .softOverlay {
      opacity: 0.9;
    }

    .tooltip {
      position: fixed;
      z-index: 100;
      padding: 1rem;
      margin: 1rem;

      background: white;
      border: 1px solid rgb(230, 230, 230);
      box-shadow: 0 0.5rem 1rem 0 rgba(0, 0, 0, 0.2);
    }
  `;
P([
  d({ type: Object })
], y.prototype, "page", 2);
P([
  d({ type: Array })
], y.prototype, "items", 2);
P([
  d({ type: Boolean })
], y.prototype, "pagenumbers", 2);
P([
  d()
], y.prototype, "itemPart", 2);
P([
  d()
], y.prototype, "itemStyle", 2);
P([
  d()
], y.prototype, "onClickItem", 2);
P([
  d({ type: Boolean })
], y.prototype, "backdrop", 2);
P([
  d({ type: Object })
], y.prototype, "overlay", 2);
P([
  d({ type: Object })
], y.prototype, "tooltip", 2);
P([
  d({ type: Object })
], y.prototype, "trace", 2);
y = P([
  v("docling-img-page")
], y);
/**
 * @license
 * Copyright 2017 Google LLC
 * SPDX-License-Identifier: BSD-3-Clause
 */
const qe = Symbol();
class Qt {
  get taskComplete() {
    return this.t || (this.i === 1 ? this.t = new Promise((t, e) => {
      this.o = t, this.h = e;
    }) : this.i === 3 ? this.t = Promise.reject(this.l) : this.t = Promise.resolve(this.u)), this.t;
  }
  constructor(t, e, s) {
    var o;
    this.p = 0, this.i = 0, (this._ = t).addController(this);
    const r = typeof e == "object" ? e : { task: e, args: s };
    this.v = r.task, this.j = r.args, this.m = r.argsEqual ?? We, this.k = r.onComplete, this.A = r.onError, this.autoRun = r.autoRun ?? !0, "initialValue" in r && (this.u = r.initialValue, this.i = 2, this.O = (o = this.T) == null ? void 0 : o.call(this));
  }
  hostUpdate() {
    this.autoRun === !0 && this.S();
  }
  hostUpdated() {
    this.autoRun === "afterUpdate" && this.S();
  }
  T() {
    if (this.j === void 0) return;
    const t = this.j();
    if (!Array.isArray(t)) throw Error("The args function must return an array");
    return t;
  }
  async S() {
    const t = this.T(), e = this.O;
    this.O = t, t === e || t === void 0 || e !== void 0 && this.m(e, t) || await this.run(t);
  }
  async run(t) {
    var n, h, a, l, p;
    let e, s;
    t ?? (t = this.T()), this.O = t, this.i === 1 ? (n = this.q) == null || n.abort() : (this.t = void 0, this.o = void 0, this.h = void 0), this.i = 1, this.autoRun === "afterUpdate" ? queueMicrotask(() => this._.requestUpdate()) : this._.requestUpdate();
    const r = ++this.p;
    this.q = new AbortController();
    let o = !1;
    try {
      e = await this.v(t, { signal: this.q.signal });
    } catch (c) {
      o = !0, s = c;
    }
    if (this.p === r) {
      if (e === qe) this.i = 0;
      else {
        if (o === !1) {
          try {
            (h = this.k) == null || h.call(this, e);
          } catch {
          }
          this.i = 2, (a = this.o) == null || a.call(this, e);
        } else {
          try {
            (l = this.A) == null || l.call(this, s);
          } catch {
          }
          this.i = 3, (p = this.h) == null || p.call(this, s);
        }
        this.u = e, this.l = s;
      }
      this._.requestUpdate();
    }
  }
  abort(t) {
    var e;
    this.i === 1 && ((e = this.q) == null || e.abort(t));
  }
  get value() {
    return this.u;
  }
  get error() {
    return this.l;
  }
  get status() {
    return this.i;
  }
  render(t) {
    var e, s, r, o;
    switch (this.i) {
      case 0:
        return (e = t.initial) == null ? void 0 : e.call(t);
      case 1:
        return (s = t.pending) == null ? void 0 : s.call(t);
      case 2:
        return (r = t.complete) == null ? void 0 : r.call(t, this.value);
      case 3:
        return (o = t.error) == null ? void 0 : o.call(t, this.error);
      default:
        throw Error("Unexpected status: " + this.i);
    }
  }
}
const We = (i, t) => i === t || i.length === t.length && i.every((e, s) => !ct(e, t[s]));
function Ve(i) {
  return Object.values((i == null ? void 0 : i.pages) ?? {}).sort(
    (t, e) => t.page_no - e.page_no
  );
}
async function te(i, t = {}) {
  var a, l, p;
  let e;
  typeof i == "string" ? e = await (await fetch(i)).json() : typeof i == "object" && b.Document(i) && (e = i);
  const s = Ve(e), r = Array.from(
    xe(e, { traversePictures: !0 })
  ).map(([c]) => c);
  let o = r;
  const n = /* @__PURE__ */ new Set();
  if (typeof t.items == "string") {
    const c = new Set(
      t.items.split(",").map((u) => u.trim()).filter((u) => u.length > 0)
    );
    if (c.size > 0) {
      o = [];
      for (const u of r) {
        const g = u.self_ref.split("/"), _ = [];
        for (let $ = 2; $ < g.length + 1; $++)
          _.push(g.slice(0, $).join("/"));
        (a = u.prov) == null || a.forEach(($) => _.push(`#/pages/${$.page_no}`)), _.some(($) => c.has($)) ? o.push(u) : (l = u.prov) == null || l.forEach(($) => n.add($.page_no));
      }
    }
  } else if (Array.isArray(t.items)) {
    o = [];
    const c = new Set(t.items);
    for (const u of r)
      c.has(u) ? o.push(u) : (p = u.prov) == null || p.forEach((g) => n.add(g.page_no));
  }
  const h = {};
  for (const c of s)
    h[c.page_no] = [];
  for (const c of o)
    for (const u of c.prov ?? [])
      h[u.page_no].at(-1) !== c && h[u.page_no].push(c);
  return s.map((c) => ({
    page: c,
    items: h[c.page_no],
    trimmed: n.has(c.page_no)
  }));
}
var Fe = Object.defineProperty, Ge = Object.getOwnPropertyDescriptor, It = (i, t, e, s) => {
  for (var r = s > 1 ? void 0 : s ? Ge(t, e) : t, o = i.length - 1, n; o >= 0; o--)
    (n = i[o]) && (r = (s ? n(t, e, r) : n(r)) || r);
  return s && r && Fe(t, e, r), r;
};
let F = class extends A {
  render() {
    if (this.page && this.items && this.items.length > 0) {
      const { page_no: i, size: t } = this.page, { width: e = 1, height: s = 1 } = t, r = this.items.map((n) => {
        var a, l;
        const h = q((l = (a = n.prov) == null ? void 0 : a.find((p) => p.page_no === i)) == null ? void 0 : l.bbox, this.page);
        return [
          (h.l + h.r) / 2,
          (h.t + h.b) / 2
        ];
      }), o = `M${r[0][0]} 0 ${r.slice(0).map((n) => `L${n[0]} ${n[1]}`)} L${r.at(-1)[0]} ${s}`;
      return m`<svg viewBox="0 0 ${e} ${s}">
        <marker
          id="arrow"
          viewBox="0 0 10 10"
          refX="0"
          refY="5"
          markerWidth="4"
          markerHeight="4"
          orient="auto"
        >
          <path d="M 0 0 L 10 5 L 0 10" />
        </marker>

        <path
          class="halo"
          vector-effect="non-scaling-stroke"
          d=${o}
          marker-start="url(#arrow)"
        />

        <path
          class="trace"
          vector-effect="non-scaling-stroke"
          d=${o}
          marker-start="url(#arrow)"
        />
      </svg>`;
    }
  }
};
F.styles = O`
    svg {
      position: absolute;
      inset: 0;
      pointer-events: none;
    }

    .halo {
      fill: none;
      stroke-width: 5px;
      stroke: white;
      stroke-linejoin: round;
    }

    .trace {
      fill: none;
      stroke-width: 3px;
      stroke: rgba(0, 0, 255);
      stroke-linejoin: round;
    }

    marker > path {
      fill: rgba(0, 0, 255);
      stroke: none;
    }
  `;
It([
  d({ type: Object })
], F.prototype, "page", 2);
It([
  d({ type: Array })
], F.prototype, "items", 2);
F = It([
  v("docling-trace")
], F);
var Ke = Object.defineProperty, Je = Object.getOwnPropertyDescriptor, E = (i, t, e, s) => {
  for (var r = s > 1 ? void 0 : s ? Je(t, e) : t, o = i.length - 1, n; o >= 0; o--)
    (n = i[o]) && (r = (s ? n(t, e, r) : n(r)) || r);
  return s && r && Ke(t, e, r), r;
};
let w = class extends A {
  constructor() {
    super(...arguments), this.src = "", this.trim = "pages", this.fetchTask = new Qt(this, {
      task: async ([i, t]) => te(i, { items: t }),
      args: () => [this.src, this.items]
    });
  }
  render() {
    const i = Array.from(this.childNodes);
    return this.fetchTask.render({
      pending: () => m`<p>...</p>`,
      complete: (t) => m`
        ${t.filter((e) => !this.trim || e.items.length > 0).map(
        ({ page: e, items: s }) => m`<docling-img-page
                part="page"
                .page=${e}
                .items=${s}
                .pagenumbers=${this.pagenumbers !== void 0}
                .itemPart=${this.itemPart}
                .itemStyle=${this.itemStyle}
                .onClickItem=${this.onClickItem}
                .backdrop=${this.backdrop !== void 0}
                .overlay=${i.find((r) => r instanceof wt)}
                .tooltip=${i.find((r) => r instanceof At)}
                .trace=${i.find((r) => r instanceof F)}
              />`
      )}
      `
    });
  }
};
w.styles = O`
    :host {
      width: fit-content;

      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 2px;

      color: black;
    }
  `;
E([
  d()
], w.prototype, "alt", 2);
E([
  d()
], w.prototype, "backdrop", 2);
E([
  d()
], w.prototype, "items", 2);
E([
  d({ type: Boolean })
], w.prototype, "pagenumbers", 2);
E([
  d()
], w.prototype, "src", 2);
E([
  d()
], w.prototype, "trim", 2);
E([
  d()
], w.prototype, "itemPart", 2);
E([
  d()
], w.prototype, "itemStyle", 2);
E([
  d()
], w.prototype, "onClickItem", 2);
w = E([
  v("docling-img")
], w);
/**
 * @license
 * Copyright 2018 Google LLC
 * SPDX-License-Identifier: BSD-3-Clause
 */
const Ze = (i) => i ?? f;
var Xe = Object.defineProperty, Ye = Object.getOwnPropertyDescriptor, k = (i, t, e, s) => {
  for (var r = s > 1 ? void 0 : s ? Ye(t, e) : t, o = i.length - 1, n; o >= 0; o--)
    (n = i[o]) && (r = (s ? n(t, e, r) : n(r)) || r);
  return s && r && Xe(t, e, r), r;
};
let x = class extends A {
  constructor() {
    super(...arguments), this.columns = [];
  }
  render() {
    var i, t, e, s, r, o, n;
    return m`
      ${this.pagenumbers ? m`
            <tr>
              <td
                part="page-number-top"
                class="page-number"
                title="Page ${(i = this.page) == null ? void 0 : i.page_no}"
                colspan=${((t = this.columns) == null ? void 0 : t.length) ?? 1}
              >
                ${(e = this.page) == null ? void 0 : e.page_no}
              </td>
            </tr>
          ` : f}
      ${(s = this.items) == null ? void 0 : s.map(
      (h) => {
        var a;
        return m`<tr
            part=${"item" + (this.itemPart ? " " + this.itemPart(this.page, h) : "")}
            style=${Ze((a = this.itemStyle) == null ? void 0 : a.call(this, this.page, h))}
          >
            ${this.columns.map((l) => {
          const p = l.cloneNode(!0);
          return p.item = h, p.page = this.page, m`
                <td
                  @onclick=${(c) => {
            var u;
            return (u = this.onClickItem) == null ? void 0 : u.call(this, c, this.page, h);
          }}
                >
                  ${p}
                </td>
              `;
        })}
          </tr>`;
      }
    )}
      ${this.pagenumbers ? m`<tr>
            <td
              part="page-number-bottom"
              class="page-number"
              title="Page ${(r = this.page) == null ? void 0 : r.page_no}"
              colspan=${((o = this.columns) == null ? void 0 : o.length) ?? 1}
            >
              ${(n = this.page) == null ? void 0 : n.page_no}
            </td>
          </tr>` : f}
    `;
  }
};
x.styles = O`
    :host {
      display: table-row-group;
    }

    tbody {
      border-bottom: 1px solid rgb(220, 220, 220);
    }

    .page-number {
      padding: 0.25rem;

      background-color: white;
      color: rgb(120, 120, 120);
      font-size: 0.75rem;
      line-height: 1rem;
    }

    tr {
      cursor: pointer;
    }

    tr:not(:nth-last-child(2)) {
      border-bottom: 1px dotted rgb(220, 220, 220);
    }

    td {
      padding: 1rem;
      background-color: white;
      vertical-align: top;
    }

    tr:hover td {
      filter: brightness(95%);
    }
  `;
k([
  d({ type: Object })
], x.prototype, "page", 2);
k([
  d({ type: Array })
], x.prototype, "items", 2);
k([
  d({ type: Array })
], x.prototype, "columns", 2);
k([
  d({ type: Boolean })
], x.prototype, "pagenumbers", 2);
k([
  d()
], x.prototype, "itemPart", 2);
k([
  d()
], x.prototype, "itemStyle", 2);
k([
  d()
], x.prototype, "onClickItem", 2);
x = k([
  v("docling-table-page")
], x);
var Qe = Object.getOwnPropertyDescriptor, tr = (i, t, e, s) => {
  for (var r = s > 1 ? void 0 : s ? Qe(t, e) : t, o = i.length - 1, n; o >= 0; o--)
    (n = i[o]) && (r = n(r) || r);
  return r;
};
let lt = class extends tt {
  constructor() {
    super(...arguments), this.type = "column";
  }
};
lt = tr([
  v("docling-column")
], lt);
var er = Object.defineProperty, rr = Object.getOwnPropertyDescriptor, U = (i, t, e, s) => {
  for (var r = s > 1 ? void 0 : s ? rr(t, e) : t, o = i.length - 1, n; o >= 0; o--)
    (n = i[o]) && (r = (s ? n(t, e, r) : n(r)) || r);
  return s && r && er(t, e, r), r;
};
let C = class extends A {
  constructor() {
    super(...arguments), this.src = "", this.fetchTask = new Qt(this, {
      task: async ([i, t]) => te(i, { items: t }),
      args: () => [this.src, this.items]
    });
  }
  render() {
    const i = Array.from(this.childNodes).filter(
      (e) => e.nodeName.toLowerCase().startsWith("docling-column")
    ), t = i.length > 0 ? i : [new lt(), new lt()];
    return i.length === 0 && t[1].appendChild(new ht()), this.fetchTask.render({
      pending: () => m`<p>...</p>`,
      complete: (e) => m`
        <table part="pages">
          ${e.filter((s) => s.items.length > 0).map(
        ({ page: s, items: r }) => m`<docling-table-page
                  .page=${s}
                  .items=${r}
                  .columns=${t}
                  .pagenumbers=${this.pagenumbers !== void 0}
                  .itemPart=${this.itemPart}
                  .itemStyle=${this.itemStyle}
                  .onClickItem=${this.onClickItem}
                />`
      )}
        </table>
      `
    });
  }
};
C.styles = O`
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      background-color: white;
    }
  `;
U([
  d()
], C.prototype, "alt", 2);
U([
  d()
], C.prototype, "items", 2);
U([
  d({ type: Boolean })
], C.prototype, "pagenumbers", 2);
U([
  d()
], C.prototype, "src", 2);
U([
  d()
], C.prototype, "itemPart", 2);
U([
  d()
], C.prototype, "itemStyle", 2);
U([
  d()
], C.prototype, "onClickItem", 2);
C = U([
  v("docling-table")
], C);
export {
  at as AnnotationPictureClassification,
  _t as AnnotationPictureDescription,
  z as DoclingAnnotationElement,
  D as DoclingItemElement,
  y as ImgPage,
  w as ImgPages,
  F as ImgTrace,
  wt as ItemOverlay,
  ht as ItemProvenance,
  vt as ItemTable,
  Lt as ItemTemplate,
  bt as ItemText,
  At as ItemTooltip,
  tt as ItemView,
  x as TablePage,
  C as TablePages,
  Vt as customDoclingAnnotationElement,
  Yt as customDoclingItemElement,
  q as normalBbox
};

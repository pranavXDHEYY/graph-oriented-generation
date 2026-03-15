# SRM Phase 2: Honest Assessment After Skeptical Review

## The Skeptical Questions

**What actually works?** Which results are reproducible and trustworthy vs which are artifacts of the rubric or skeleton pre-loading?

**What are we measuring?** String matching across concatenated output ≠ validated per-file correctness for multi-file tasks.

---

## Easy Task: ✓ SOLID (Fully Trusted)

**Result:** SRM PASS 5/5, RAG/GOG PARTIAL 3/5

**Why it's trustworthy:**
- Single file (authStore.ts) — no multi-file complexity
- Rubric checks are per-operation: field name, type, default value, action mutation
- Response is small and fully visible
- The 0.5B model genuinely produces correct Pinia syntax only when given symbolic spec

**Verdict:** This is the core SRM result. No skeleton pre-loading. The model renders correct structured code from symbolic specification. **FULLY VALIDATED.**

---

## Medium Task: ⚠️ PROMISING (Needs Deeper Inspection)

**Result:** SRM PASS 5/5, RAG PASS 5/5, GOG PASS 5/5

**Why it's less clear:**
- All three tiers are passing, which is suspicious
- Vue component skeleton pre-loads `handleLogout` and `const auth = useAuthStore()`
- Rubric checks for presence of `<template>`, `<script`, `@click`, `useAuthStore` — but these are partly in the skeleton
- The LLM's job is to add the Logout button and wire the click handler, but the skeleton makes it ambiguous what the LLM added vs what was pre-loaded

**What SRM response actually shows:**
```vue
<script setup lang="ts">
import { useAuthStore } from '../stores/authStore';
const auth = useAuthStore();
const logout = () => { auth.logout(); };
</script>
<template>
  <div>
    <span>{{ auth.user.role }}</span>
    <button @click="logout">Logout</button>
  </div>
</template>
```

The LLM produced syntactically correct, properly wired Vue code. The binding and imports are right. **This result is probably real**, but the fact that all three tiers pass suggests the rubric is loose.

**Verdict:** Likely valid, but needs explicit verification that the LLM output and skeleton-pre-loaded content are properly distinguished. **NEEDS DEEPER INSPECTION.**

---

## Hard Task: ✗ NOT TRUSTWORTHY (Needs Fix)

**Result:** SRM PASS 5/5, RAG PARTIAL 4/5, GOG FAIL 2/5

**Critical Issues:**

### Issue 1: Skeleton Pre-Loading Masks LLM Output

UserSettings.vue skeleton comes pre-loaded with:
```vue
<script setup lang="ts">
const handleLogout = () => { auth.logout(); };
const handleDelete = () => { auth.deleteUser(); };
</script>
```

The rubric checks for:
- ✓ "Delete button" — finds `<button @click="Delete">` in skeleton AND in response
- ✓ "deleteUser action" — finds in authStore
- ✓ "deleteAccount function" — finds in api_client
- ✓ "posts to /delete" — finds in api_client
- ✓ "No api_client import in Vue" — true, but irrelevant since binding is pre-loaded

**The problem:** The script block is NOT in the rendered response. The LLM produced incomplete Vue code (missing `<script>`), but the rubric passes it because the skeleton already has the handler logic. The model didn't actually render what it was asked to render.

### Issue 2: Tier 1 RAG Should Fail the Constraint

RAG response imports `api_client` into the Vue component, which should violate the `FORBIDDEN_IMPORT` constraint. But the new constraint checking code treats it lenient if the Vue block can't be extracted from RAG's single-block response.

**Current result:** RAG PARTIAL 4/5 (passes when it should fail the constraint)

### Issue 3: Tier 2 GOG Has Legitimate Failures

GOG response:
```typescript
export default class UserSettings extends Button {
  constructor() {
    super({
      name: 'Delete Account',
      onClick() { this.$emit('deleteUser'); },
      action: authStore.actions.deleteUser,
    });
  }
}
```

This is syntactically wrong for Vue (trying to extend Button class, using Vue 2 syntax). The rubric correctly scores it FAIL. **This is probably right.**

### Issue 4: The Per-File Extraction Still Has Gaps

With improved extraction, we can identify files by the `# ─── filename ───` markers. But:
- RAG/GOG don't use this format (they return single blocks)
- Fall-back heuristics for file detection are imperfect
- When extraction fails, we give a warning but don't invalidate the score

**Example:** RAG's response is a single ```typescript block with no file markers. The extraction can't cleanly separate Vue from other code.

---

## Honest Summary

| Task | Status | Why |
|------|--------|-----|
| **Easy** | ✓ PROVEN | Single-file, no skeleton tricks, syntax-correct outputs, clear metrics |
| **Medium** | ⚠️ LIKELY VALID | Syntax looks correct, but skeleton pre-loading makes it hard to verify LLM did all the work |
| **Hard** | ✗ NOT TRUSTWORTHY | SRM response incomplete (missing Vue script), skeleton masks rendering quality, rubric is coarse |

---

## What Needs to Happen Before Publishing

### For Easy Task:
✓ Already done. Document and move to paper.

### For Medium Task:
1. **Inspect skeleton and response diff** — What content is truly from LLM vs pre-loaded?
2. **Run with empty skeleton** — Force LLM to produce complete Vue component from scratch
3. **Validate syntax** — Run generated code through a TypeScript/Vue parser

### For Hard Task:
1. **Fix skeleton** — Don't pre-load handlers; require LLM to render them
2. **Implement proper per-file grading** — Each file must be syntactically complete
3. **Strengthen the constraint check** — Must verify no import statement in Vue file, not just absence of certain strings
4. **Re-run after fixes** — Get true results without skeleton shortcuts

---

## The Real SRM Hypothesis

The claim is: **Symbolic specs let small LLMs render structured code correctly.**

- **Easy task proves this:** The 0.5B model produces correct Pinia mutations from symbolic spec
- **Medium task suggests it:** Code structure looks right, but skeleton ambiguity muddies the water
- **Hard task contradicts it:** The model produces incomplete code (missing Vue script block), which wouldn't work in production

The Hard task failure might mean:
1. **Hypothesis is wrong for multi-file tasks** — Too much cognitive load even with per-file rendering
2. **Architecture needs adjustment** — Maybe three separate LLM calls aren't enough; need inter-file context
3. **Skeleton is doing the work** — Pre-loaded logic makes it appear to work when it doesn't

---

## Recommendation

**Don't declare victory yet.** The Easy task is rock-solid. Medium needs verification. Hard needs a rebuild.

For a paper submission:
- **Lead with Easy task** — This is your strong result
- **Acknowledge Medium ambiguity** — Show you tested it, note the skeleton issue
- **Fix Hard before submitting** — Or acknowledge it as future work

The SRM hypothesis is not disproven. Easy task proves the core idea. But the multi-file case needs more rigorous validation.

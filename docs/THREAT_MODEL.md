# Threat model

**Attacker controls** the *content* of untrusted channels: web pages, emails, third-party tool output, files,
other agents, and (when principals are authenticated) other users of a shared memory.
**Attacker does not control:** channel metadata (host-authenticated), the store's code/storage, the owner's
own turns, the system prompt.
**Goal:** a consequential tool call, or a corrupted answer, in a later session.

| Mechanism | Defended | Notes |
|---|---|---|
| Explicit flow: untrusted memory value reaches a tool argument | yes | `Policy(balanced)` |
| Laundering through summarise / merge / reflect | yes, conservatively | `span` attribution errors can only add taint; citations can only add taint |
| Implicit flow: untrusted memory changes *which* trusted memory is retrieved, or *whether* the agent acts | only by `Policy(strict)` | strict gates any consequential call while untrusted spans are in context; costs confirmation burden |
| Query-only attacker (MINJA) acting as a normal user | only with authenticated principals | `ChannelRegistry(trusted_users=...)`. Without it the attacker's turns are labelled TRUSTED (tested: `test_minja_needs_principal_authentication`) |
| Attacker who copies a *trusted* identifier into untrusted text | yes by default | both spans anchor on the identifier; `anchor_dominates=True` is NOT safe against reference-resolution laundering (benchmark family g: 6.7% laundering in the toy setup) |
| Tampering with stored labels at rest | detected | HMAC over each record; key management is the host's job |
| Social-engineering the user into endorsing a memory | NOT defended | endorsement is bound to a span id and an authenticated user event, and logged, but a convinced user can still upgrade an attacker's fact |
| Compromised trusted tool / compromised host | out of scope | |
| Answer-bias from untrusted memory on non-consequential answers | out of scope | provenance tags are shown to the model, not enforced |

Known weak points of the implementation: sentence attribution is lexical (hashed/IDF cosine + exact
identifiers); heavy paraphrase defeats it, in which case it falls back to the conservative label (safe, costly).
Real-LLM behaviour is **untested**.

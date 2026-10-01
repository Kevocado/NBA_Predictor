# Phase 1 — instant block, NBA

Real Chromium driving this branch's dev server, against a local fixture API on
`127.0.0.1:8020` whose shapes come from `frontend/src/api/client.ts` and whose
numbers are the same fixtures the tests use — so a shot cannot disagree with the
suite. Nothing in the page is mocked. Desktop is 1280×900, mobile is 390×844,
both at `deviceScaleFactor: 2`.

| file | state |
|---|---|
| `*-1-pretip-block-verdict-record-promise.png` | pre-tip, before the button: block, verdict, bar, record, promise line |
| `*-2-after-button-figures-once.png` | after pressing the button: prose added, each figure still once |
| `*-3-rebuilt-badge-not-counted.png` | rebuilt pick: badge, "not counted", old `PregamePick` prose absent |
| `*-4-accented-bar-segment.png` | the block close up: the pick's segment accented, the other neutral |

The accent is visible as a colour, not only as markup. Read from the live page,
`getComputedStyle` on the two segments of the moneyline bar:

```
["rgb(255, 122, 26)", "rgb(180, 189, 202)"]
   accent (BOS 62%)     neutral (MIA 38%)
```

## Regenerating

The dev server already proxies `/games`, `/games/:id`, `/hub/players`,
`/hub/track-record` and `/api/explain/nba/:id` to `127.0.0.1:8020`, so a local
API on that port is all it needs:

```bash
cd frontend && npm run dev
```

# How to submit (GitHub) — quick start

You downloaded this from Moodle and unzipped it. **This `doc-agent-starter/` folder is your project.**
You need **Git installed** (`git --version` to check) and a **GitHub account**.

The commands below run in a **terminal, inside this folder**:
```
cd path/to/doc-agent-starter      # the unzipped folder (has README.md, src/, forms/)
```

### Step 1 — Create the empty repo ON github.com first (browser)
`git` runs on your computer; `git init/add/commit` are all **local** and never touch GitHub. Only `git push`
uploads — and it can only push to a repo that **already exists** on GitHub. So make it first:
- On github.com: **New repository** → name it `doc-agent-<your team id>` → **Public** (REQUIRED — a private
  repo cannot be graded = no submission) → **do NOT** add a README, .gitignore, or license (this folder has them).

### Step 2 — Turn this folder into a repo (local)
```
git init
git add .
git commit -m "initial commit"
git branch -M main
```

### Step 3 — Connect it to GitHub and push (upload)
```
git remote add origin https://github.com/<you>/doc-agent-<team-id>.git
git push -u origin main
```
> **If `git push` asks for a password and fails:** GitHub no longer accepts your account password over HTTPS.
> Use a **Personal Access Token** as the password (github.com → Settings → Developer settings → Personal
> access tokens → generate one with `repo` scope), or install **GitHub Desktop**, or set up **SSH keys**.

> **2026 cohort:** A1 was submitted on Moodle before this stub existed, so your **first tag is
> `a2-submit`**. Your A2 push must also include the A1 artifacts you could not produce then —
> `configs/task.yaml`, `data/provenance.md` + your corpus, `notebooks/eda.ipynb`,
> `grading_kit/manifest.yaml`, and `grading_kit/heldout_pages/` + `labels.jsonl`. They are graded
> at A2. See the A2 form's "What to hand in".

### Step 4 — Add teammates & send the URL
- Repo → Settings → Collaborators → add your two teammates.
- Send the instructor the repo URL (same repo for all four milestones).

### Each milestone — submit by tagging
Fill `forms/AN_form.docx` + your `grading_kit/`, commit, then:
```
git add -A && git commit -m "A2 submission"
git tag a2-submit
git push origin main --tags        # <-- the push + tag is your submission
```
Deadlines are read from **GitHub's servers**, not your computer's clock.

Full details: `handbook/02-How-To-Submit.pdf`. Start reading at `handbook/01-START-HERE.pdf`.

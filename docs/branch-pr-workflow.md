# Branch and PR Workflow

This document outlines the branch naming conventions and PR workflow for this repository.

## Branch Naming

### Feature Branches

Use the following format for feature branches:

```
feature/<type>-<short-description>-YYYY-MM-DD
```

**Types:**
- `rl` - Reinforcement learning changes
- `sft` - Supervised fine-tuning changes
- `sim` - Simulation/infrastructure changes
- `docs` - Documentation changes
- `survey` - Research/survey work
- `daily` - Daily pipeline work (one per day)

**Examples:**
```
feature/rl-waypoint-refinement       # Long-running feature
feature/docs-roadmap-ppo             # Documentation update
feature/survey-gaia-2                # Survey work
feature/daily-2026-02-15             # Daily pipeline PR
```

### Survey Branches

Survey branches should be self-contained and deleted after merging:

```
survey-<topic>-YYYY-MM-DD
```

**Example:**
```
survey-gaia-2-2026-02-15
```

### Release Branches

```
release/v<major>.<minor>
```

## Pull Request Workflow

### 1. Create Branch

```bash
git checkout main
git pull origin main
git checkout -b feature/<type>-<short-description>
```

### 2. Make Changes

- Write clear commit messages
- Keep commits atomic
- Run tests before pushing

### 3. Push and Create PR

```bash
git push origin feature/<type>-<short-description>
```

Visit the GitHub URL to create a PR. Fill in the PR template.

### 4. Review

- Address review comments
- Squash commits if requested
- Ensure CI passes

### 5. Merge

This repository uses **squash merging**. All commits will be collapsed into one.

After merging:
- Delete the feature branch (locally and remotely)
- Update CHANGELOG.md if significant

## Merge Strategy

**Squash and Merge** is the default strategy:

- Each PR becomes one commit
- Keeps history linear
- Easier to roll back
- Commit message = PR title + description

## Best Practices

1. **Small PRs**: Keep PRs under 400 lines when possible
2. **One focus**: Each PR should do one thing
3. **Update CHANGELOG.md**: For user-facing changes
4. **Link issues**: Reference related issues
5. **Draft PRs**: Use draft PRs for work-in-progress

## Branch Protection

The following rules are configured in GitHub:

- ✅ Require PR before merge
- ✅ Require review approval (1 reviewer)
- ✅ Require up-to-date branches
- ✅ Require squash merging

## Labels

Use GitHub labels to categorize PRs:

**By Type:**
- `type: rl`
- `type: sft`
- `type: sim`
- `type: docs`
- `type: survey`

**By Status:**
- `status: ready`
- `status: review`
- `status: draft`
- `status: changes-requested`

**By Priority:**
- `priority: high`
- `priority: medium`
- `priority: low`

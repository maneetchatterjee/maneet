# Troubleshooting: "Run Workflow" Button Not Visible

If you cannot see the "Run workflow" button in GitHub Actions, follow these steps:

## Quick Fix Steps

### Step 1: Ensure You're on the Correct Branch
The workflow file is on the `copilot/implement-qlstm-on-nasa-dataset` branch.

**Option A: Navigate directly to the workflow**
1. Click this link: https://github.com/maneetchatterjee/maneet/actions/workflows/run-real-data-experiments.yml
2. You should see the workflow page with a "Run workflow" button on the right side

**Option B: Use Actions tab**
1. Go to: https://github.com/maneetchatterjee/maneet/actions
2. Make sure you're viewing actions for the `copilot/implement-qlstm-on-nasa-dataset` branch (check branch selector at top)
3. In the left sidebar, click "Run QLSTM Experiments on Real NASA Data"
4. The "Run workflow" button should appear on the right

### Step 2: Check Branch Selection
If you still don't see the button:
1. Look at the top of the GitHub page
2. Make sure the branch dropdown shows: `copilot/implement-qlstm-on-nasa-dataset`
3. If not, switch to that branch
4. Then go back to Actions tab

### Step 3: Enable GitHub Actions (if disabled)
1. Go to: https://github.com/maneetchatterjee/maneet/settings/actions
2. Under "Actions permissions", ensure actions are enabled
3. Select "Allow all actions and reusable workflows"
4. Save changes

### Step 4: Check Workflow Permissions
1. Go to: https://github.com/maneetchatterjee/maneet/settings/actions
2. Scroll down to "Workflow permissions"
3. Ensure "Read and write permissions" is selected
4. Check "Allow GitHub Actions to create and approve pull requests"
5. Save changes

## Alternative: Direct Workflow Dispatch URL

If the button still doesn't appear, use this direct URL to manually trigger the workflow:

```
https://github.com/maneetchatterjee/maneet/actions/workflows/run-real-data-experiments.yml
```

Then click the gray "Run workflow" button that appears on the right side of the page.

## Alternative: Merge to Main Branch

If nothing else works, the workflow will automatically be accessible after merging this PR to the main branch:

1. Merge the PR: `copilot/implement-qlstm-on-nasa-dataset` → `main`
2. Go to: https://github.com/maneetchatterjee/maneet/actions
3. The workflow should now be visible and runnable from the main branch

## Visual Guide

When everything is correct, you should see:

```
GitHub Actions Page
├── Left Sidebar
│   └── "Run QLSTM Experiments on Real NASA Data" ← Click here
└── Right Side
    └── [Run workflow ▼] ← This button should be visible
        ├── Branch: copilot/implement-qlstm-on-nasa-dataset
        └── [Run workflow] ← Click to execute
```

## Common Issues

### Issue 1: "This workflow has a workflow_dispatch event trigger"
This message means the workflow is correctly configured. The button should be visible on the right.

### Issue 2: Branch mismatch
The workflow file must exist on the branch you're viewing. Switch to `copilot/implement-qlstm-on-nasa-dataset`.

### Issue 3: First-time workflow
GitHub Actions may need a few moments after the workflow file is first pushed. Wait 1-2 minutes and refresh.

### Issue 4: Repository permissions
Make sure you have write access to the repository. Only repository members with write access can trigger workflows manually.

## Need Help?

If none of these steps work:

1. **Check workflow syntax**: The file at `.github/workflows/run-real-data-experiments.yml` should be valid YAML
2. **Check GitHub status**: Visit https://www.githubstatus.com/ to ensure Actions are operational
3. **Clear browser cache**: Sometimes GitHub UI caches can cause issues
4. **Try different browser**: Use an incognito window or different browser

## Success Indicators

You'll know it's working when:
- ✅ You can see the workflow in the Actions tab left sidebar
- ✅ The "Run workflow" button is visible on the right
- ✅ Clicking the button shows a dropdown with branch selection
- ✅ After clicking "Run workflow", a new workflow run appears in the list

## After Triggering

Once the workflow runs:
1. You'll see a new entry in the workflow runs list
2. Click on it to watch progress (takes ~20-30 minutes)
3. When complete, scroll down to "Artifacts" section
4. Download `experiment-results` to see your NASA data results!

---

**Still stuck?** Reply with a screenshot of your Actions page and I can provide more specific guidance.

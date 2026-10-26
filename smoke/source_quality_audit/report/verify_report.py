"""Browser verification of offline loading, figures, filters and evidence views."""
import csv
import io
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(ROOT / ".browsers")
os.environ["LD_LIBRARY_PATH"] = str(ROOT / ".browser-libs/extracted/usr/lib/x86_64-linux-gnu")
os.environ["TMPDIR"] = str(ROOT / ".tmp")
from playwright.sync_api import sync_playwright


def main():
    checks = {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True,args=["--no-sandbox"])
        page = browser.new_page(viewport={"width":1500,"height":1100},device_scale_factor=1)
        errors, remote = [], []
        page.on("pageerror",lambda e: errors.append(str(e)))
        page.on("request",lambda r: remote.append(r.url) if r.url.startswith(("http://","https://")) else None)
        page.goto((ROOT / "index.html").as_uri(),wait_until="load")
        page.wait_for_function("window.auditReady === true",timeout=60000)
        assert page.evaluate("window.auditRecordCount") == 3400
        assert page.locator("#sourceTable tbody tr").count() == 34
        assert page.locator("#reviewCards .review-card").count() == 20
        assert not page.evaluate("document.documentElement.scrollWidth > window.innerWidth")
        checks["desktop_load_all_records"] = True
        page.screenshot(path=str(ROOT / "preview-desktop.png"))
        page.locator("#sourceSearch").fill("OpenMathInstruct")
        assert page.locator("#sourceTable tbody tr:visible").count() == 1
        page.locator("#sourceTable [data-open-source]").filter(has_text="OpenMathInstruct").click()
        assert "100 matching" in page.locator("#resultCount").inner_text()
        page.locator("#reviewOutcome").select_option("major_defect")
        assert "51 matching" in page.locator("#resultCount").inner_text()
        page.locator("#reviewDimension").select_option("prompt_quality")
        subset = int(page.locator("#resultCount").inner_text().split()[0].replace(",",""))
        assert 0 < subset <= 51
        checks["source_outcome_dimension_filters"] = True
        page.locator("#resetFilters").click()
        page.locator("#reviewSearch").fill("chairs in the office canteen")
        page.wait_for_timeout(350)
        assert "1 matching" in page.locator("#resultCount").inner_text()
        page.locator("#reviewCards [data-open-record]").click()
        assert page.locator("#recordDialog").is_visible()
        assert "02c64d27af7ed4e46b84cef9" in page.locator("#dialogBody").inner_text()
        assert "26" in page.locator("#dialogBody").inner_text()
        assert page.locator("#dialogBody .record-text").count() == 4
        page.screenshot(path=str(ROOT / "preview-record.png"))
        page.keyboard.press("Escape")
        assert not page.locator("#recordDialog").is_visible()
        checks["search_full_record_and_keyboard_close"] = True
        page.locator("#resetFilters").click()
        page.locator("#nextPage").click()
        assert "Page 2" in page.locator("#pageNumber").inner_text()
        page.locator("#previousPage").click()
        checks["pagination"] = True
        page.locator("#sourceSearch").fill("")
        page.locator("#sourceSort").select_option("major")
        assert "nvidia/Nemotron-SWE-v1" in page.locator("#sourceTable tbody tr").first.inner_text()
        with page.expect_download() as dl:
            page.locator("#downloadSources").click()
        download = dl.value
        download.save_as(str(ROOT / "browser-source-summary.csv"))
        with (ROOT / "browser-source-summary.csv").open() as stream:
            assert len(list(csv.DictReader(stream))) == 34
        checks["source_sort_and_csv_download"] = True
        page.locator("#sources").scroll_into_view_if_needed()
        page.screenshot(path=str(ROOT / "preview-sources.png"))
        page.evaluate("window.scrollTo(0,0)")
        page.set_viewport_size({"width":390,"height":844})
        page.wait_for_timeout(300)
        assert not page.evaluate("document.documentElement.scrollWidth > window.innerWidth")
        page.screenshot(path=str(ROOT / "preview-mobile.png"))
        page.locator('[data-open-record="2989839b6172fc5e223d25cb"]').click()
        assert page.locator("#recordDialog").is_visible()
        assert not page.locator("#recordDialog").evaluate("e=>e.scrollWidth > e.clientWidth + 1")
        page.locator("#closeDialog").click()
        checks["mobile_layout_and_dialog"] = True
        assert not errors,errors
        assert not remote,remote
        checks["javascript_errors"] = errors
        checks["external_network_requests"] = remote
        browser.close()
    (ROOT / "verification.json").write_text(json.dumps(checks,indent=2)+"\n")
    print(json.dumps(checks,indent=2))


if __name__ == "__main__":
    main()

"""Check the offline report against frozen reviews in desktop and mobile browsers.

Uses the local Playwright/Chromium installation from the earlier SFT audit.
Override PLAYWRIGHT_BROWSERS_PATH or LD_LIBRARY_PATH for another installation.
All new outputs and temporary browser profiles stay in this report directory.
"""
import csv
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1] / 'source_quality_audit' / 'report'
os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH', str(BASE / '.browsers'))
os.environ.setdefault('LD_LIBRARY_PATH', str(BASE / '.browser-libs/extracted/usr/lib/x86_64-linux-gnu'))
(HERE / '.tmp').mkdir(exist_ok=True)
os.environ['TMPDIR'] = str(HERE / '.tmp')
from playwright.sync_api import sync_playwright


def main():
    reviews = [json.loads(line) for line in (HERE.parent/'verdicts.jsonl').read_text().splitlines()]
    checks = {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=['--no-sandbox'])
        page = browser.new_page(viewport={'width':1500,'height':1100},device_scale_factor=1)
        errors, remote = [], []
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('request',lambda r:remote.append(r.url) if r.url.startswith(('http://','https://')) else None)
        page.goto((HERE/'index.html').as_uri(),wait_until='load',timeout=120000)
        page.wait_for_function('() => window.auditReady === true',timeout=120000)
        assert page.evaluate('window.auditRecordCount') == 3200
        assert page.locator('#sourceTable tbody tr').count() == 16
        assert page.locator('#reviewCards .review-card').count() == 20
        assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
        checks['offline_load_3200_records_16_sources'] = True
        page.screenshot(path=str(HERE/'preview-desktop.png'))
        page.locator('#sourceSearch').fill('math-decontam')
        assert page.locator('#sourceTable tbody tr:visible').count() == 1
        page.locator('#sourceTable [data-open-source]').filter(has_text='math-decontam').click()
        assert '200 matching' in page.locator('#resultCount').inner_text()
        page.locator('#reviewOutcome').select_option('major_defect')
        assert '184 matching' in page.locator('#resultCount').inner_text()
        page.locator('#reviewDimension').select_option('answer_correctness')
        expected = sum('math-decontam' in r['source'] and r['overall']=='major_defect' and r['dimensions']['answer_correctness'] in ['major_issue','minor_issue','uncertain'] for r in reviews)
        assert f'{expected} matching' in page.locator('#resultCount').inner_text()
        checks['source_outcome_dimension_filters'] = True
        page.locator('#resetFilters').click()
        aid = '021cd16244bb0bb5b6456084'
        page.locator('#reviewSearch').fill(aid)
        page.wait_for_timeout(400)
        assert '1 matching' in page.locator('#resultCount').inner_text()
        page.locator('#reviewCards [data-open-record]').click()
        assert page.locator('#recordDialog').is_visible()
        assert aid in page.locator('#dialogBody').inner_text()
        expected_text = page.evaluate('(id)=>lookup.get(id).record.messages.map(m=>m.content)',aid)
        actual_text = page.locator('#dialogBody .message > details:first-of-type > pre').all_text_contents()
        assert actual_text == expected_text
        assert page.locator('#dialogBody .message').count() == 2
        page.screenshot(path=str(HERE/'preview-record.png'))
        page.keyboard.press('Escape')
        checks['search_exact_original_messages_and_keyboard_close'] = True
        page.locator('#resetFilters').click()
        page.locator('#nextPage').click()
        assert 'Page 2' in page.locator('#pageNumber').inner_text()
        page.locator('#previousPage').click()
        checks['pagination'] = True
        page.locator('#sourceSearch').fill('')
        page.locator('#sourceSort').select_option('major')
        assert 'math-decontam' in page.locator('#sourceTable tbody tr').first.inner_text()
        with page.expect_download() as download:
            page.locator('#downloadSources').click()
        download.value.save_as(str(HERE/'browser-source-summary.csv'))
        with (HERE/'browser-source-summary.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 16 and sum(int(row['major_count']) for row in rows) == 1337
        assert all(row['sampled']=='200' for row in rows)
        checks['source_sort_and_csv_totals'] = True
        page.locator('#sources').scroll_into_view_if_needed()
        page.screenshot(path=str(HERE/'preview-sources.png'))
        page.evaluate('window.scrollTo(0,0)')
        page.set_viewport_size({'width':390,'height':844})
        page.wait_for_timeout(300)
        assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
        page.screenshot(path=str(HERE/'preview-mobile.png'))
        multi = page.evaluate('data.reviews.find(r=>r.record.messages.filter(m=>m.role==="assistant").length===20).audit_id')
        page.evaluate('(id)=>openRecord(id)',multi)
        assert page.locator('#recordDialog').is_visible()
        assert not page.locator('#recordDialog').evaluate('e=>e.scrollWidth>e.clientWidth+1')
        original = page.evaluate('(id)=>lookup.get(id).record.messages.map(m=>m.content)',multi)
        assert page.locator('#dialogBody .message > details:first-of-type > pre').all_text_contents() == original
        assert len(original) >= 40
        page.locator('#closeDialog').click()
        checks['mobile_layout_and_full_20_assistant_turn_conversation'] = True
        assert not errors,errors
        assert not remote,remote
        checks['javascript_errors'] = errors
        checks['external_network_requests'] = remote
        browser.close()
    (HERE/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__ == '__main__':
    main()

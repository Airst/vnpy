from data_manager.ts_downloader.concept_manager import ConceptManager

if __name__ == "__main__":
    manager = ConceptManager()
    manager.download_daily()
    manager.download_members()